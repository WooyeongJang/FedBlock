package com.fedblock.client.ui.features.training

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
import com.fedblock.client.domain.model.ClientStatus
import com.fedblock.client.domain.model.TrainingType

/**
 * 연합학습 훈련 화면
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun TrainingScreen(
    viewModel: TrainingViewModel = hiltViewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // 헤더
        Text(
            text = "Federated Learning Training",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )
        
        // 클라이언트 상태 카드
        ClientStatusCard(
            status = uiState.clientStatus,
            clientId = uiState.clientId
        )
        
        // 훈련 설정 카드
        TrainingConfigCard(
            trainingType = uiState.selectedTrainingType,
            config = uiState.config,
            onTrainingTypeChange = viewModel::updateTrainingType,
            onConfigChange = viewModel::updateConfig
        )
        
        // 훈련 진행 상황
        if (uiState.clientStatus == ClientStatus.TRAINING) {
            TrainingProgressCard(
                progress = uiState.trainingProgress
            )
        }
        
        // 액션 버튼들
        ActionButtons(
            clientStatus = uiState.clientStatus,
            onStartTraining = viewModel::startTraining,
            onStopTraining = viewModel::stopTraining,
            onRegisterClient = viewModel::registerClient,
            onUnregisterClient = viewModel::unregisterClient
        )
        
        // 훈련 히스토리
        TrainingHistoryCard(
            trainingHistory = uiState.trainingHistory
        )
        
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
private fun ClientStatusCard(
    status: ClientStatus,
    clientId: String
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Client Status",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(text = "Client ID: $clientId")
                
                StatusChip(status = status)
            }
        }
    }
}

@Composable
private fun StatusChip(status: ClientStatus) {
    val (color, icon) = when (status) {
        ClientStatus.IDLE -> MaterialTheme.colorScheme.primary to Icons.Default.CheckCircle
        ClientStatus.TRAINING -> MaterialTheme.colorScheme.secondary to Icons.Default.ModelTraining
        ClientStatus.UPLOADING -> MaterialTheme.colorScheme.tertiary to Icons.Default.CloudUpload
        ClientStatus.ERROR -> MaterialTheme.colorScheme.error to Icons.Default.Error
    }
    
    AssistChip(
        onClick = { },
        label = { Text(status.name) },
        leadingIcon = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = color
            )
        }
    )
}

@Composable
private fun TrainingConfigCard(
    trainingType: TrainingType,
    config: com.fedblock.client.domain.model.FederatedLearningConfig,
    onTrainingTypeChange: (TrainingType) -> Unit,
    onConfigChange: (com.fedblock.client.domain.model.FederatedLearningConfig) -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Training Configuration",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            // 훈련 타입 선택
            var expanded by remember { mutableStateOf(false) }
            
            ExposedDropdownMenuBox(
                expanded = expanded,
                onExpandedChange = { expanded = it }
            ) {
                OutlinedTextField(
                    value = trainingType.name,
                    onValueChange = { },
                    readOnly = true,
                    label = { Text("Training Type") },
                    trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                    modifier = Modifier
                        .fillMaxWidth()
                        .menuAnchor()
                )
                
                ExposedDropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    TrainingType.values().forEach { type ->
                        DropdownMenuItem(
                            text = { Text(type.name) },
                            onClick = {
                                onTrainingTypeChange(type)
                                expanded = false
                            }
                        )
                    }
                }
            }
            
            // 학습률, 에포크, 배치 크기 설정
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedTextField(
                    value = config.learningRate.toString(),
                    onValueChange = { value ->
                        value.toDoubleOrNull()?.let { lr ->
                            onConfigChange(config.copy(learningRate = lr))
                        }
                    },
                    label = { Text("Learning Rate") },
                    modifier = Modifier.weight(1f)
                )
                
                OutlinedTextField(
                    value = config.epochs.toString(),
                    onValueChange = { value ->
                        value.toIntOrNull()?.let { epochs ->
                            onConfigChange(config.copy(epochs = epochs))
                        }
                    },
                    label = { Text("Epochs") },
                    modifier = Modifier.weight(1f)
                )
                
                OutlinedTextField(
                    value = config.batchSize.toString(),
                    onValueChange = { value ->
                        value.toIntOrNull()?.let { batchSize ->
                            onConfigChange(config.copy(batchSize = batchSize))
                        }
                    },
                    label = { Text("Batch Size") },
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
private fun TrainingProgressCard(
    progress: com.fedblock.client.domain.repository.TrainingProgress
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Training Progress",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Text(text = "Epoch ${progress.currentEpoch} / ${progress.totalEpochs}")
            
            LinearProgressIndicator(
                progress = progress.currentEpoch.toFloat() / progress.totalEpochs.toFloat(),
                modifier = Modifier.fillMaxWidth()
            )
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(text = "Loss: %.4f".format(progress.currentLoss))
                Text(text = "Accuracy: %.2f%%".format(progress.currentAccuracy * 100))
            }
            
            if (progress.estimatedTimeRemaining > 0) {
                Text(
                    text = "Estimated time remaining: ${progress.estimatedTimeRemaining / 1000}s",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}

@Composable
private fun ActionButtons(
    clientStatus: ClientStatus,
    onStartTraining: () -> Unit,
    onStopTraining: () -> Unit,
    onRegisterClient: () -> Unit,
    onUnregisterClient: () -> Unit
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onStartTraining,
                enabled = clientStatus == ClientStatus.IDLE,
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Start Training")
            }
            
            Button(
                onClick = onStopTraining,
                enabled = clientStatus == ClientStatus.TRAINING,
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.Stop, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Stop Training")
            }
        }
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedButton(
                onClick = onRegisterClient,
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.PersonAdd, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Register")
            }
            
            OutlinedButton(
                onClick = onUnregisterClient,
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.PersonRemove, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Unregister")
            }
        }
    }
}

@Composable
private fun TrainingHistoryCard(
    trainingHistory: List<com.fedblock.client.domain.model.TrainingResult>
) {
    Card(
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Training History",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            if (trainingHistory.isEmpty()) {
                Text(
                    text = "No training history available",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                LazyColumn(
                    modifier = Modifier.heightIn(max = 200.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(trainingHistory.take(5)) { result ->
                        TrainingHistoryItem(result = result)
                    }
                }
            }
        }
    }
}

@Composable
private fun TrainingHistoryItem(
    result: com.fedblock.client.domain.model.TrainingResult
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Round ${result.modelParameters.round}",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "${result.trainingTime / 1000}s",
                    style = MaterialTheme.typography.bodySmall
                )
            }
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Accuracy: %.2f%%".format(result.accuracy * 100),
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = "Loss: %.4f".format(result.loss),
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}
