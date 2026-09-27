package com.retirementreadinesslab.entitlements

import android.content.Context

/** Debug-only entitlement controls used by local and closed-test builds. */
class DefaultEntitlementProvider(
    @Suppress("UNUSED_PARAMETER") context: Context? = null
) : ProEntitlementProvider {
    override val providerName: String = "Fake debug entitlement provider"
    override val allowsDeveloperOverrides: Boolean = true

    override suspend fun currentEntitlement(storedLocalUnlock: Boolean): ProEntitlement {
        return ProEntitlement(isProUnlocked = storedLocalUnlock)
    }

    override suspend fun redeemPromoCode(
        activity: android.app.Activity?,
        promoCode: String,
        storedLocalUnlock: Boolean
    ): ProEntitlement {
        return if (promoCode.trim() == DEBUG_PROMO_CODE) {
            ProEntitlement(
                isProUnlocked = true,
                message = "Debug Pro promo code accepted."
            )
        } else {
            ProEntitlement(
                isProUnlocked = storedLocalUnlock,
                message = "Debug Pro promo code was not recognized.",
                shouldPersist = false
            )
        }
    }

    override suspend fun setDeveloperOverride(
        isProUnlocked: Boolean,
        storedLocalUnlock: Boolean
    ): ProEntitlement {
        return ProEntitlement(
            isProUnlocked = isProUnlocked,
            message = if (isProUnlocked) {
                "Debug Pro unlock enabled."
            } else {
                "Debug Pro unlock reset to Free."
            }
        )
    }

    private companion object {
        const val DEBUG_PROMO_CODE = "proUnlockForTesting"
    }
}
