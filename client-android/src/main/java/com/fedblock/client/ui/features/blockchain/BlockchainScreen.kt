package com.fedblock.client.ui.features.blockchain

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * 블록체인 상태 화면
 */
@Composable
fun BlockchainScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // 헤더
        Text(
            text = "Blockchain Status",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )
        
        // 연결 상태 카드
        ConnectionStatusCard()
        
        // 블록체인 정보 카드
        BlockchainInfoCard()
        
        // 스마트 컨트랙트 정보 카드
        SmartContractCard()
    }
}

@Composable
private fun ConnectionStatusCard() {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Network Connection",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Text("Connected to Monad Testnet")
                }
                
                AssistChip(
                    onClick = { },
                    label = { Text("Online") }
                )
            }
        }
    }
}

@Composable
private fun BlockchainInfoCard() {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Blockchain Information",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            InfoRow(label = "Network", value = "Monad Testnet")
            InfoRow(label = "Chain ID", value = "41144")
            InfoRow(label = "RPC URL", value = "https://testnet-rpc.monad.xyz")
            InfoRow(label = "Block Height", value = "2,451,832")
            InfoRow(label = "Gas Price", value = "20 Gwei")
        }
    }
}

@Composable
private fun SmartContractCard() {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Smart Contract",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            InfoRow(
                label = "Contract Address", 
                value = "0x7b1d8fB9De56669BA8F38Eba759d53526364774F"
            )
            InfoRow(label = "Contract Name", value = "FederatedLearningAudit")
            InfoRow(label = "Version", value = "1.0.0")
            InfoRow(label = "Events Logged", value = "1,247")
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = { /* View on explorer */ },
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.Launch, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("View on Explorer")
                }
                
                OutlinedButton(
                    onClick = { /* Test connection */ },
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(Icons.Default.NetworkCheck, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Test Connection")
                }
            }
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Medium
        )
    }
}
