package com.retirementreadinesslab.entitlements

import android.app.Activity

data class ProEntitlement(
    val isProUnlocked: Boolean,
    val message: String? = null,
    val shouldPersist: Boolean = true
)

interface ProEntitlementProvider {
    val providerName: String
    val allowsDeveloperOverrides: Boolean
    val supportsUserPurchases: Boolean
        get() = false

    suspend fun currentEntitlement(storedLocalUnlock: Boolean): ProEntitlement

    suspend fun purchasePro(
        activity: Activity,
        storedLocalUnlock: Boolean
    ): ProEntitlement {
        return ProEntitlement(
            isProUnlocked = storedLocalUnlock,
            message = "Pro purchase is not available in this build.",
            shouldPersist = false
        )
    }

    suspend fun redeemPromoCode(
        activity: Activity?,
        promoCode: String,
        storedLocalUnlock: Boolean
    ): ProEntitlement {
        return ProEntitlement(
            isProUnlocked = storedLocalUnlock,
            message = "Promo code unlock is not available in this build.",
            shouldPersist = false
        )
    }

    suspend fun setDeveloperOverride(
        isProUnlocked: Boolean,
        storedLocalUnlock: Boolean
    ): ProEntitlement
}
