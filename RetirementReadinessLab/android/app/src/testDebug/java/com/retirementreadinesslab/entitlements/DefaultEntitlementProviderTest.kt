package com.retirementreadinesslab.entitlements

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DefaultEntitlementProviderTest {
    @Test
    fun debugProviderAllowsDeveloperOverrides() = runBlocking {
        val provider = DefaultEntitlementProvider()

        assertEquals("Fake debug entitlement provider", provider.providerName)
        assertTrue(provider.allowsDeveloperOverrides)
        assertFalse(provider.currentEntitlement(storedLocalUnlock = false).isProUnlocked)
        assertTrue(provider.currentEntitlement(storedLocalUnlock = true).isProUnlocked)
    }

    @Test
    fun debugProviderReturnsRequestedOverrideState() = runBlocking {
        val provider = DefaultEntitlementProvider()

        val pro = provider.setDeveloperOverride(
            isProUnlocked = true,
            storedLocalUnlock = false
        )
        val free = provider.setDeveloperOverride(
            isProUnlocked = false,
            storedLocalUnlock = true
        )

        assertTrue(pro.isProUnlocked)
        assertEquals("Debug Pro unlock enabled.", pro.message)
        assertFalse(free.isProUnlocked)
        assertEquals("Debug Pro unlock reset to Free.", free.message)
    }

    @Test
    fun debugProviderAcceptsTestingPromoCode() = runBlocking {
        val provider = DefaultEntitlementProvider()

        val entitlement = provider.redeemPromoCode(
            activity = null,
            promoCode = "proUnlockForTesting",
            storedLocalUnlock = false
        )

        assertTrue(entitlement.isProUnlocked)
        assertTrue(entitlement.shouldPersist)
        assertEquals("Debug Pro promo code accepted.", entitlement.message)
    }

    @Test
    fun debugProviderRejectsUnknownPromoCode() = runBlocking {
        val provider = DefaultEntitlementProvider()

        val entitlement = provider.redeemPromoCode(
            activity = null,
            promoCode = "unknown",
            storedLocalUnlock = false
        )

        assertFalse(entitlement.isProUnlocked)
        assertFalse(entitlement.shouldPersist)
        assertEquals("Debug Pro promo code was not recognized.", entitlement.message)
    }
}
