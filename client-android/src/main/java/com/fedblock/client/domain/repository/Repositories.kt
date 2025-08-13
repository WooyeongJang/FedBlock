package com.fedblock.client.domain.repository

import com.fedblock.client.domain.model.*
import kotlinx.coroutines.flow.Flow

/**
 * 연합학습 저장소 인터페이스
 */
interface FederatedLearningRepository {
    
    /**
     * 서버에 클라이언트 등록
     */
    suspend fun registerClient(clientId: String): Result<Boolean>
    
    /**
     * 서버에서 클라이언트 해제
     */
    suspend fun unregisterClient(clientId: String): Result<Boolean>
    
    /**
     * 글로벌 모델 파라미터 가져오기
     */
    suspend fun getGlobalModelParameters(trainingType: TrainingType): Result<ModelParameters>
    
    /**
     * 로컬 모델 훈련 수행
     */
    suspend fun trainLocalModel(
        config: FederatedLearningConfig,
        initialParams: ModelParameters
    ): Result<TrainingResult>
    
    /**
     * 훈련된 모델 파라미터 서버에 업로드
     */
    suspend fun uploadModelParameters(
        trainingResult: TrainingResult,
        trainingType: TrainingType
    ): Result<Boolean>
    
    /**
     * 피어 클라이언트들과 모델 파라미터 교환 (Gossip 학습용)
     */
    suspend fun exchangeParametersWithPeers(
        modelParams: ModelParameters,
        peers: List<String>
    ): Result<List<ModelParameters>>
    
    /**
     * 클라이언트 상태 관찰
     */
    fun observeClientStatus(): Flow<ClientStatus>
    
    /**
     * 훈련 진행 상황 관찰
     */
    fun observeTrainingProgress(): Flow<TrainingProgress>
}

/**
 * 블록체인 감사 저장소 인터페이스
 */
interface BlockchainAuditRepository {
    
    /**
     * 감사 로그를 블록체인에 기록
     */
    suspend fun logAuditEvent(event: AuditLogEntry): Result<TransactionResult>
    
    /**
     * 블록체인에서 감사 로그 조회
     */
    suspend fun getAuditLogs(clientId: String): Result<List<AuditLogEntry>>
    
    /**
     * 모델 파라미터의 무결성 검증
     */
    suspend fun verifyModelIntegrity(
        modelParams: ModelParameters,
        expectedHash: String
    ): Result<Boolean>
    
    /**
     * 블록체인 연결 상태 확인
     */
    suspend fun checkBlockchainConnection(): Result<Boolean>
}

/**
 * 훈련 진행 상황
 */
data class TrainingProgress(
    val currentEpoch: Int,
    val totalEpochs: Int,
    val currentLoss: Double,
    val currentAccuracy: Double,
    val estimatedTimeRemaining: Long
)
