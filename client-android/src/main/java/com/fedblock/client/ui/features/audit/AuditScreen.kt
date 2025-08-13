package com.fedblock.client.ui.features.audit

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.fedblock.client.domain.model.AuditEventType
import com.fedblock.client.domain.model.AuditLogEntry
import java.text.SimpleDateFormat
import java.util.*

/**
 * 감사 로그 화면
 */
@Composable
fun AuditScreen(
    viewModel: AuditViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    LaunchedEffect(Unit) {
        viewModel.loadAuditLogs()
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // 헤더
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Blockchain Audit Logs",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold
            )
            
            IconButton(onClick = { viewModel.loadAuditLogs() }) {
                Icon(Icons.Default.Refresh, contentDescription = "Refresh")
            }
        }
        
        // 통계 카드
        AuditStatsCard(auditLogs = uiState.auditLogs)
        
        // 감사 로그 목록
        if (uiState.isLoading) {
            Box(
                modifier = Modifier.fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator()
            }
        } else if (uiState.auditLogs.isEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "No audit logs available",
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        } else {
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(uiState.auditLogs) { auditLog ->
                    AuditLogCard(auditLog = auditLog)
                }
            }
        }
        
        // 에러 메시지
        uiState.errorMessage?.let { error ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer
                )
            ) {
                Text(
                    text = error,
                    modifier = Modifier.padding(16.dp),
                    color = MaterialTheme.colorScheme.onErrorContainer
                )
            }
        }
    }
}

@Composable
private fun AuditStatsCard(auditLogs: List<AuditLogEntry>) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Audit Statistics",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(12.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatItem(
                    label = "Total Events",
                    value = auditLogs.size.toString()
                )
                
                StatItem(
                    label = "Training Events",
                    value = auditLogs.count { 
                        it.eventType == AuditEventType.TRAINING_START || 
                        it.eventType == AuditEventType.TRAINING_COMPLETE 
                    }.toString()
                )
                
                StatItem(
                    label = "Model Updates",
                    value = auditLogs.count { 
                        it.eventType == AuditEventType.MODEL_UPDATE 
                    }.toString()
                )
            }
        }
    }
}

@Composable
private fun StatItem(label: String, value: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

@Composable
private fun AuditLogCard(auditLog: AuditLogEntry) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                EventTypeChip(eventType = auditLog.eventType)
                
                Text(
                    text = formatTimestamp(auditLog.timestamp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            
            Text(
                text = "Client: ${auditLog.clientId}",
                style = MaterialTheme.typography.bodyMedium
            )
            
            auditLog.transactionHash?.let { hash ->
                Text(
                    text = "Tx: ${hash.take(10)}...${hash.takeLast(10)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary
                )
            }
            
            if (auditLog.data.isNotEmpty()) {
                Column(
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = "Data:",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Bold
                    )
                    
                    auditLog.data.forEach { (key, value) ->
                        Text(
                            text = "$key: $value",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun EventTypeChip(eventType: AuditEventType) {
    val (color, icon) = when (eventType) {
        AuditEventType.CLIENT_REGISTRATION -> MaterialTheme.colorScheme.primary to Icons.Default.PersonAdd
        AuditEventType.CLIENT_DEREGISTRATION -> MaterialTheme.colorScheme.secondary to Icons.Default.PersonRemove
        AuditEventType.TRAINING_START -> MaterialTheme.colorScheme.tertiary to Icons.Default.PlayArrow
        AuditEventType.TRAINING_COMPLETE -> MaterialTheme.colorScheme.primary to Icons.Default.CheckCircle
        AuditEventType.MODEL_UPDATE -> MaterialTheme.colorScheme.secondary to Icons.Default.CloudUpload
        AuditEventType.PARAMETER_EXCHANGE -> MaterialTheme.colorScheme.tertiary to Icons.Default.SwapHoriz
    }
    
    AssistChip(
        onClick = { },
        label = { Text(eventType.name.replace("_", " ")) },
        leadingIcon = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color
            )
        }
    )
}

private fun formatTimestamp(timestamp: Long): String {
    val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    return formatter.format(Date(timestamp))
}
