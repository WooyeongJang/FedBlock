package com.fedblock.client.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.fedblock.client.R
import com.fedblock.client.ui.features.training.TrainingScreen
import com.fedblock.client.ui.features.audit.AuditScreen
import com.fedblock.client.ui.features.blockchain.BlockchainScreen
import com.fedblock.client.ui.features.settings.SettingsScreen
import com.fedblock.client.ui.theme.FedBlockTheme
import dagger.hilt.android.AndroidEntryPoint

/**
 * FedBlock 메인 액티비티
 * 
 * 연합학습, 감사 로그, 블록체인, 설정 화면을 
 * 네비게이션으로 관리합니다.
 */
@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setContent {
            FedBlockTheme {
                FedBlockApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FedBlockApp() {
    val navController = rememberNavController()
    
    val items = listOf(
        NavigationItem(
            title = "Training",
            icon = Icons.Filled.ModelTraining,
            route = "training"
        ),
        NavigationItem(
            title = "Audit",
            icon = Icons.Filled.Security,
            route = "audit"
        ),
        NavigationItem(
            title = "Blockchain",
            icon = Icons.Filled.AccountBalanceWallet,
            route = "blockchain"
        ),
        NavigationItem(
            title = "Settings",
            icon = Icons.Filled.Settings,
            route = "settings"
        )
    )
    
    Scaffold(
        modifier = Modifier.fillMaxSize(),
        bottomBar = {
            NavigationBar {
                val navBackStackEntry by navController.currentBackStackEntryAsState()
                val currentDestination = navBackStackEntry?.destination
                
                items.forEach { item ->
                    NavigationBarItem(
                        icon = { Icon(item.icon, contentDescription = item.title) },
                        label = { Text(item.title) },
                        selected = currentDestination?.hierarchy?.any { 
                            it.route == item.route 
                        } == true,
                        onClick = {
                            navController.navigate(item.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = "training",
            modifier = Modifier.padding(innerPadding)
        ) {
            composable("training") { TrainingScreen() }
            composable("audit") { AuditScreen() }
            composable("blockchain") { BlockchainScreen() }
            composable("settings") { SettingsScreen() }
        }
    }
}

data class NavigationItem(
    val title: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector,
    val route: String
)
