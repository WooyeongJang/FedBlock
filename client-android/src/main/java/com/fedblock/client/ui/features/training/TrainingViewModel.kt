package com.fedblock.client.ui.features.training

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.fedblock.client.domain.model.*
import com.fedblock.client.domain.repository.FederatedLearningRepository
import com.fedblock.client.domain.repository.BlockchainAuditRepository
import com.fedblock.client.domain.repository.TrainingProgress
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

/**
 * 훈련 화면 ViewModel
 */
@HiltViewModel
class TrainingViewModel @Inject constructor(
    private val federatedLearningRepository: FederatedLearningRepository,
    private val blockchainAuditRepository: BlockchainAuditRepository
) : ViewModel() {
    
    private val _uiState = MutableStateFlow(TrainingUiState())
    val uiState: StateFlow<TrainingUiState> = combine(
        _uiState,
        federatedLearningRepository.observeClientStatus(),
        federatedLearningRepository.observeTrainingProgress()
    ) { state, clientStatus, trainingProgress ->
        state.copy(
            clientStatus = clientStatus,
            trainingProgress = trainingProgress
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = TrainingUiState()
    )
    
    /**
     * 훈련 타입 업데이트
     */
    fun updateTrainingType(trainingType: TrainingType) {
        _uiState.update { currentState ->
            currentState.copy(
                selectedTrainingType = trainingType,
                config = currentState.config.copy(trainingType = trainingType)
            )
        }
    }
    
    /**
     * 훈련 설정 업데이트
     */
    fun updateConfig(config: FederatedLearningConfig) {
        _uiState.update { currentState ->
            currentState.copy(config = config)
        }
    }
    
    /**
     * 클라이언트 등록
     */
    fun registerClient() {
        viewModelScope.launch {
            try {
                val clientId = _uiState.value.clientId
                val result = federatedLearningRepository.registerClient(clientId)
                
                if (result.isSuccess) {
                    // 블록체인에 등록 이벤트 기록
                    logAuditEvent(
                        eventType = AuditEventType.CLIENT_REGISTRATION,
                        data = mapOf("client_id" to clientId)
                    )
                    
                    _uiState.update { it.copy(errorMessage = null) }
                } else {
                    _uiState.update { 
                        it.copy(errorMessage = "Registration failed: ${result.exceptionOrNull()?.message}")
                    }
                }
            } catch (e: Exception) {
                _uiState.update { 
                    it.copy(errorMessage = "Registration error: ${e.message}")
                }
            }
        }
    }
    
    /**
     * 클라이언트 해제
     */
    fun unregisterClient() {
        viewModelScope.launch {
            try {
                val clientId = _uiState.value.clientId
                val result = federatedLearningRepository.unregisterClient(clientId)
                
                if (result.isSuccess) {
                    // 블록체인에 해제 이벤트 기록
                    logAuditEvent(
                        eventType = AuditEventType.CLIENT_DEREGISTRATION,
                        data = mapOf("client_id" to clientId)
                    )
                    
                    _uiState.update { it.copy(errorMessage = null) }
                } else {
                    _uiState.update { 
                        it.copy(errorMessage = "Unregistration failed: ${result.exceptionOrNull()?.message}")
                    }
                }
            } catch (e: Exception) {
                _uiState.update { 
                    it.copy(errorMessage = "Unregistration error: ${e.message}")
                }
            }
        }
    }
    
    /**
     * 훈련 시작
     */
    fun startTraining() {
        viewModelScope.launch {
            try {
                val currentState = _uiState.value
                
                // 글로벌 모델 파라미터 가져오기
                val globalParamsResult = federatedLearningRepository.getGlobalModelParameters(
                    currentState.selectedTrainingType
                )
                
                if (globalParamsResult.isFailure) {
                    _uiState.update { 
                        it.copy(errorMessage = "Failed to get global parameters: ${globalParamsResult.exceptionOrNull()?.message}")
                    }
                    return@launch
                }
                
                val globalParams = globalParamsResult.getOrNull()!!
                val initialParams = globalParams.copy(clientId = currentState.clientId)
                
                // 블록체인에 훈련 시작 이벤트 기록
                logAuditEvent(
                    eventType = AuditEventType.TRAINING_START,
                    data = mapOf(
                        "client_id" to currentState.clientId,
                        "training_type" to currentState.selectedTrainingType.name,
                        "round" to globalParams.round.toString()
                    )
                )
                
                // 로컬 모델 훈련 수행
                val trainingResult = federatedLearningRepository.trainLocalModel(
                    config = currentState.config,
                    initialParams = initialParams
                )
                
                if (trainingResult.isSuccess) {
                    val result = trainingResult.getOrNull()!!
                    
                    // 훈련 히스토리에 추가
                    _uiState.update { state ->
                        state.copy(
                            trainingHistory = state.trainingHistory + result,
                            errorMessage = null
                        )
                    }
                    
                    // 훈련된 모델 파라미터 업로드
                    uploadTrainingResult(result)
                    
                    // 블록체인에 훈련 완료 이벤트 기록
                    logAuditEvent(
                        eventType = AuditEventType.TRAINING_COMPLETE,
                        data = mapOf(
                            "client_id" to currentState.clientId,
                            "accuracy" to result.accuracy.toString(),
                            "loss" to result.loss.toString(),
                            "training_time" to result.trainingTime.toString()
                        )
                    )
                    
                } else {
                    _uiState.update { 
                        it.copy(errorMessage = "Training failed: ${trainingResult.exceptionOrNull()?.message}")
                    }
                }
                
            } catch (e: Exception) {
                _uiState.update { 
                    it.copy(errorMessage = "Training error: ${e.message}")
                }
            }
        }
    }
    
    /**
     * 훈련 중지
     */
    fun stopTraining() {
        // 훈련 중지 로직 구현
        // 현재는 간단히 상태만 변경
        _uiState.update { it.copy(errorMessage = "Training stopped by user") }
    }
    
    /**
     * 훈련 결과 업로드
     */
    private suspend fun uploadTrainingResult(result: TrainingResult) {
        try {
            val uploadResult = federatedLearningRepository.uploadModelParameters(
                trainingResult = result,
                trainingType = _uiState.value.selectedTrainingType
            )
            
            if (uploadResult.isSuccess) {
                // 블록체인에 모델 업데이트 이벤트 기록
                logAuditEvent(
                    eventType = AuditEventType.MODEL_UPDATE,
                    data = mapOf(
                        "client_id" to result.clientId,
                        "round" to result.modelParameters.round.toString()
                    )
                )
            }
        } catch (e: Exception) {
            _uiState.update { 
                it.copy(errorMessage = "Upload error: ${e.message}")
            }
        }
    }
    
    /**
     * 블록체인에 감사 이벤트 기록
     */
    private suspend fun logAuditEvent(
        eventType: AuditEventType,
        data: Map<String, String>
    ) {
        try {
            val auditEvent = AuditLogEntry(
                id = UUID.randomUUID().toString(),
                eventType = eventType,
                clientId = _uiState.value.clientId,
                timestamp = System.currentTimeMillis(),
                data = data
            )
            
            blockchainAuditRepository.logAuditEvent(auditEvent)
        } catch (e: Exception) {
            // 감사 로그 실패는 메인 로직에 영향을 주지 않음
            println("Failed to log audit event: ${e.message}")
        }
    }
}

/**
 * 훈련 화면 UI 상태
 */
data class TrainingUiState(
    val clientId: String = UUID.randomUUID().toString().take(8),
    val clientStatus: ClientStatus = ClientStatus.IDLE,
    val selectedTrainingType: TrainingType = TrainingType.MNIST,
    val config: FederatedLearningConfig = FederatedLearningConfig(),
    val trainingProgress: TrainingProgress = TrainingProgress(0, 0, 0.0, 0.0, 0L),
    val trainingHistory: List<TrainingResult> = emptyList(),
    val errorMessage: String? = null
)
