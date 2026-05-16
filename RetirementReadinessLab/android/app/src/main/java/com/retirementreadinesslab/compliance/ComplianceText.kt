package com.retirementreadinesslab.compliance

object ComplianceText {
    const val educationalDisclaimer =
        "Retirement Readiness Lab provides educational estimates based on user-entered assumptions. " +
            "It is not financial, tax, legal, or investment advice. Results are not guarantees and " +
            "should be reviewed with qualified professionals before making retirement decisions."

    const val localDataNotice =
        "Scenario data is stored locally on this device. The app does not require an account, " +
            "brokerage connection, cloud sync, or analytics transmission for the MVP."

    const val exportNotice =
        "Exports are created only after you choose a share or backup action. Shared reports and " +
            "backups can contain sensitive financial assumptions."

    const val deletionNotice =
        "Delete Local Data removes saved scenarios from this device and reloads sample data."

    const val reportPrivacyNotice =
        "This report was generated locally from user-entered scenario data. Keep exported PDFs, " +
            "text reports, CSV files, and JSON backups in a trusted location."

    val disclosureBullets = listOf(
        "Educational estimates, not advice or guarantees.",
        "No account or brokerage link is required.",
        "Scenario data stays local unless you export or share it.",
        "Exports may include sensitive financial assumptions.",
        "Federal taxes use modeled assumptions and should be reviewed before decisions."
    )
}
