package com.retirementreadinesslab

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.retirementreadinesslab.ui.theme.RetirementReadinessLabTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            RetirementReadinessLabTheme {
                RetirementReadinessLabApp()
            }
        }
    }
}
