package com.retirementreadinesslab

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Article
import androidx.compose.material.icons.automirrored.filled.ReceiptLong
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.retirementreadinesslab.state.rememberRetirementLabState
import com.retirementreadinesslab.ui.screens.BudgetScreen
import com.retirementreadinesslab.ui.screens.DashboardScreen
import com.retirementreadinesslab.ui.screens.LabScreen
import com.retirementreadinesslab.ui.screens.ReportsScreen
import com.retirementreadinesslab.ui.screens.ResultsScreen
import com.retirementreadinesslab.ui.screens.SetupScreen
import com.retirementreadinesslab.ui.screens.WelcomeScreen

private data class Destination(
    val route: String,
    val label: String,
    val icon: ImageVector
)

private val destinations = listOf(
    Destination("dashboard", "Home", Icons.Filled.Home),
    Destination("setup", "Setup", Icons.Filled.Tune),
    Destination("budget", "Budget", Icons.AutoMirrored.Filled.ReceiptLong),
    Destination("lab", "Lab", Icons.Filled.Science),
    Destination("reports", "Reports", Icons.AutoMirrored.Filled.Article)
)

@Composable
fun RetirementReadinessLabApp() {
    val navController = rememberNavController()
    val appState = rememberRetirementLabState()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route ?: "launch"
    val showBottomBar = currentRoute !in setOf("launch", "welcome")

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                NavigationBar {
                    destinations.forEach { destination ->
                        NavigationBarItem(
                            selected = currentRoute == destination.route,
                            onClick = {
                                if (currentRoute != destination.route) {
                                    navController.navigate(destination.route) {
                                        launchSingleTop = true
                                        popUpTo("dashboard")
                                    }
                                }
                            },
                            icon = {
                                Icon(
                                    imageVector = destination.icon,
                                    contentDescription = destination.label
                                )
                            },
                            label = { Text(destination.label) }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = "launch",
            modifier = Modifier.padding(innerPadding)
        ) {
            composable("launch") {
                LaunchRoute(appState) { route ->
                    navController.navigate(route) {
                        launchSingleTop = true
                        popUpTo("launch") { inclusive = true }
                    }
                }
            }
            composable("welcome") {
                WelcomeScreen(
                    state = appState,
                    onStartSetup = {
                        appState.completeFirstLaunch()
                        navController.navigate("dashboard") {
                            launchSingleTop = true
                            popUpTo("welcome") { inclusive = true }
                        }
                        navController.navigate("setup") {
                            launchSingleTop = true
                        }
                    }
                )
            }
            composable("dashboard") {
                DashboardScreen(
                    state = appState,
                    onViewResults = { navController.navigate("results") }
                )
            }
            composable("setup") {
                SetupScreen(
                    state = appState,
                    onRunCurrentSetup = {
                        navController.navigate("dashboard") {
                            launchSingleTop = true
                            popUpTo("dashboard")
                        }
                    }
                )
            }
            composable("budget") { BudgetScreen(appState) }
            composable("lab") { LabScreen(appState) }
            composable("reports") { ReportsScreen(appState) }
            composable("results") { ResultsScreen(appState) }
        }
    }
}

@Composable
private fun LaunchRoute(
    state: com.retirementreadinesslab.state.RetirementLabState,
    onReady: (String) -> Unit
) {
    LaunchedEffect(state.isLoading, state.hasCompletedFirstLaunch) {
        if (!state.isLoading) {
            onReady(if (state.hasCompletedFirstLaunch) "dashboard" else "welcome")
        }
    }

    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        CircularProgressIndicator()
    }
}
