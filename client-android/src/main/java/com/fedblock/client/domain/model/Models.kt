package com.fedblock.client.domain.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import java.math.BigInteger

/**
 * 연합학습 모델 파라미터
 */
@Parcelize
data class ModelParameters(
    val weights: List<List<Double>>,
    val bias: List<Double>,
    val round: Int = 0,
    val clientId: String = "",
    val timestamp: Long = System.currentTimeMillis()
) : Parcelable

/**
 * 연합학습 설정
 */
@Parcelize
data class FederatedLearningConfig(
    val learningRate: Double = 0.01,
    val epochs: Int = 10,
    val batchSize: Int = 32,
    val trainingType: TrainingType = TrainingType.MNIST
) : Parcelable

/**
 * 훈련 타입
 */
enum class TrainingType {
    MNIST,
    DETERMINISTIC_MNIST,
    GOSSIP_MNIST,
    CHEST_X_RAY_PNEUMONIA
}

/**
 * 클라이언트 상태
 */
enum class ClientStatus {
    IDLE,
    TRAINING,
    UPLOADING,
    ERROR
}

/**
 * 훈련 결과
 */
@Parcelize
data class TrainingResult(
    val modelParameters: ModelParameters,
    val accuracy: Double,
    val loss: Double,
    val trainingTime: Long,
    val clientId: String
) : Parcelable

/**
 * 블록체인 감사 로그 항목
 */
@Parcelize
data class AuditLogEntry(
    val id: String,
    val eventType: AuditEventType,
    val clientId: String,
    val timestamp: Long,
    val transactionHash: String? = null,
    val data: Map<String, String> = emptyMap()
) : Parcelable

/**
 * 감사 이벤트 타입
 */
enum class AuditEventType {
    CLIENT_REGISTRATION,
    CLIENT_DEREGISTRATION,
    TRAINING_START,
    TRAINING_COMPLETE,
    MODEL_UPDATE,
    PARAMETER_EXCHANGE
}

/**
 * 블록체인 트랜잭션 결과
 */
@Parcelize
data class TransactionResult(
    val hash: String,
    val success: Boolean,
    val gasUsed: BigInteger? = null,
    val blockNumber: BigInteger? = null
) : Parcelable
