package com.retirementreadinesslab.entitlements

import android.app.Activity
import android.content.Context
import com.android.billingclient.api.AcknowledgePurchaseParams
import com.android.billingclient.api.BillingClient
import com.android.billingclient.api.BillingClientStateListener
import com.android.billingclient.api.BillingFlowParams
import com.android.billingclient.api.BillingResult
import com.android.billingclient.api.PendingPurchasesParams
import com.android.billingclient.api.ProductDetails
import com.android.billingclient.api.Purchase
import com.android.billingclient.api.PurchasesUpdatedListener
import com.android.billingclient.api.QueryProductDetailsParams
import com.android.billingclient.api.QueryPurchasesParams
import com.android.billingclient.api.acknowledgePurchase
import com.android.billingclient.api.queryProductDetails
import com.android.billingclient.api.queryPurchasesAsync
import kotlin.coroutines.resume
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull

/** Google Play-backed entitlement provider used by release builds. */
class DefaultEntitlementProvider(context: Context) : ProEntitlementProvider {
    override val providerName: String = "Google Play Billing"
    override val allowsDeveloperOverrides: Boolean = false
    override val supportsUserPurchases: Boolean = true

    private val connectionMutex = Mutex()
    private var purchaseResult = CompletableDeferred<Pair<BillingResult, List<Purchase>?>>()
    private val purchasesUpdatedListener = PurchasesUpdatedListener { result, purchases ->
        purchaseResult.complete(result to purchases)
    }
    private val billingClient = BillingClient.newBuilder(context.applicationContext)
        .setListener(purchasesUpdatedListener)
        .enablePendingPurchases(
            PendingPurchasesParams.newBuilder()
                .enableOneTimeProducts()
                .build()
        )
        .enableAutoServiceReconnection()
        .build()

    override suspend fun currentEntitlement(storedLocalUnlock: Boolean): ProEntitlement {
        val setup = connect()
        if (!setup.isOk) {
            return transientFailure(storedLocalUnlock, setup)
        }

        val result = billingClient.queryPurchasesAsync(
            QueryPurchasesParams.newBuilder()
                .setProductType(BillingClient.ProductType.INAPP)
                .build()
        )
        if (!result.billingResult.isOk) {
            return transientFailure(storedLocalUnlock, result.billingResult)
        }

        val purchase = result.purchasesList.firstOrNull { it.isCompletedProPurchase() }
        if (purchase != null) {
            val acknowledged = acknowledgeIfNeeded(purchase)
            if (!acknowledged.isOk) {
                return transientFailure(storedLocalUnlock = true, acknowledged)
            }
        }
        return ProEntitlement(isProUnlocked = purchase != null)
    }

    override suspend fun purchasePro(
        activity: Activity,
        storedLocalUnlock: Boolean
    ): ProEntitlement {
        val setup = connect()
        if (!setup.isOk) return transientFailure(storedLocalUnlock, setup)

        val productDetails = queryProProduct()
            ?: return ProEntitlement(
                isProUnlocked = storedLocalUnlock,
                message = "Pro Unlock is not available from Google Play right now.",
                shouldPersist = false
            )
        val productParams = BillingFlowParams.ProductDetailsParams.newBuilder()
            .setProductDetails(productDetails)
            .apply {
                productDetails.oneTimePurchaseOfferDetailsList
                    ?.firstOrNull()
                    ?.offerToken
                    ?.takeIf { it.isNotBlank() }
                    ?.let(::setOfferToken)
            }
            .build()
        purchaseResult = CompletableDeferred()
        val launchResult = billingClient.launchBillingFlow(
            activity,
            BillingFlowParams.newBuilder()
                .setProductDetailsParamsList(listOf(productParams))
                .build()
        )
        if (!launchResult.isOk) return purchaseFailure(storedLocalUnlock, launchResult)

        val update = withTimeoutOrNull(PURCHASE_TIMEOUT_MILLIS) { purchaseResult.await() }
            ?: return ProEntitlement(
                isProUnlocked = storedLocalUnlock,
                message = "Google Play did not finish the purchase. Check Play and try Restore Purchase.",
                shouldPersist = false
            )
        val (result, purchases) = update
        if (!result.isOk) return purchaseFailure(storedLocalUnlock, result)

        val purchase = purchases?.firstOrNull { it.isCompletedProPurchase() }
            ?: return ProEntitlement(
                isProUnlocked = storedLocalUnlock,
                message = if (purchases?.any { it.purchaseState == Purchase.PurchaseState.PENDING } == true) {
                    "The Pro purchase is pending. Access will unlock after Google Play completes it."
                } else {
                    "Google Play did not return a completed Pro purchase."
                },
                shouldPersist = false
            )
        val acknowledged = acknowledgeIfNeeded(purchase)
        if (!acknowledged.isOk) return transientFailure(storedLocalUnlock = true, acknowledged)
        return ProEntitlement(isProUnlocked = true, message = "Pro unlocked.")
    }

    override suspend fun redeemPromoCode(
        activity: Activity?,
        promoCode: String,
        storedLocalUnlock: Boolean
    ): ProEntitlement {
        return ProEntitlement(
            isProUnlocked = storedLocalUnlock,
            message = "Redeem the promo code in Google Play, then tap Restore Purchase.",
            shouldPersist = false
        )
    }

    override suspend fun setDeveloperOverride(
        isProUnlocked: Boolean,
        storedLocalUnlock: Boolean
    ): ProEntitlement {
        return ProEntitlement(
            isProUnlocked = storedLocalUnlock,
            message = "Developer entitlement controls are unavailable in release builds.",
            shouldPersist = false
        )
    }

    private suspend fun queryProProduct(): ProductDetails? {
        val result = billingClient.queryProductDetails(
            QueryProductDetailsParams.newBuilder()
                .setProductList(
                    listOf(
                        QueryProductDetailsParams.Product.newBuilder()
                            .setProductId(PRO_PRODUCT_ID)
                            .setProductType(BillingClient.ProductType.INAPP)
                            .build()
                    )
                )
                .build()
        )
        if (!result.billingResult.isOk) return null
        return result.productDetailsList?.firstOrNull { it.productId == PRO_PRODUCT_ID }
    }

    private suspend fun acknowledgeIfNeeded(purchase: Purchase): BillingResult {
        if (purchase.isAcknowledged) return okResult()
        return billingClient.acknowledgePurchase(
            AcknowledgePurchaseParams.newBuilder()
                .setPurchaseToken(purchase.purchaseToken)
                .build()
        )
    }

    private suspend fun connect(): BillingResult {
        return withTimeoutOrNull(CONNECTION_TIMEOUT_MILLIS) {
            connectionMutex.withLock {
                if (billingClient.isReady) return@withLock okResult()
                suspendCancellableCoroutine { continuation ->
                    billingClient.startConnection(object : BillingClientStateListener {
                        override fun onBillingSetupFinished(result: BillingResult) {
                            if (continuation.isActive) continuation.resume(result)
                        }

                        override fun onBillingServiceDisconnected() = Unit
                    })
                }
            }
        } ?: BillingResult.newBuilder()
            .setResponseCode(BillingClient.BillingResponseCode.SERVICE_UNAVAILABLE)
            .setDebugMessage("Timed out connecting to Google Play Billing.")
            .build()
    }

    private fun Purchase.isCompletedProPurchase(): Boolean {
        return purchaseState == Purchase.PurchaseState.PURCHASED &&
            products.contains(PRO_PRODUCT_ID)
    }

    private fun transientFailure(
        storedLocalUnlock: Boolean,
        result: BillingResult
    ): ProEntitlement {
        return ProEntitlement(
            isProUnlocked = storedLocalUnlock,
            message = "Google Play purchase status could not be checked: ${result.debugMessage}",
            shouldPersist = false
        )
    }

    private fun purchaseFailure(
        storedLocalUnlock: Boolean,
        result: BillingResult
    ): ProEntitlement {
        val message = if (result.responseCode == BillingClient.BillingResponseCode.USER_CANCELED) {
            "Pro purchase canceled."
        } else {
            "Pro purchase could not start: ${result.debugMessage}"
        }
        return ProEntitlement(
            isProUnlocked = storedLocalUnlock,
            message = message,
            shouldPersist = false
        )
    }

    private fun okResult(): BillingResult {
        return BillingResult.newBuilder()
            .setResponseCode(BillingClient.BillingResponseCode.OK)
            .build()
    }

    private val BillingResult.isOk: Boolean
        get() = responseCode == BillingClient.BillingResponseCode.OK

    private companion object {
        const val PRO_PRODUCT_ID = "pro_unlock"
        const val CONNECTION_TIMEOUT_MILLIS = 15_000L
        const val PURCHASE_TIMEOUT_MILLIS = 120_000L
    }
}
