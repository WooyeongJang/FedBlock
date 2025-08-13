package com.fedblock.client.data.repository

import android.content.Context
import com.fedblock.client.data.blockchain.BlockchainService
import com.fedblock.client.domain.model.*
import com.fedblock.client.domain.repository.BlockchainAuditRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton

/**
 * 블록체인 감사 저장소 구현체
 */
@Singleton
class BlockchainAuditRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    private val blockchainService: BlockchainService
) : BlockchainAuditRepository {
    
    override suspend fun logAuditEvent(event: AuditLogEntry): Result<TransactionResult> {
        return try {
            val txResult = blockchainService.logEvent(
                eventType = event.eventType.name,
                clientId = event.clientId,
                data = event.data
            )
            Result.success(txResult)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    override suspend fun getAuditLogs(clientId: String): Result<List<AuditLogEntry>> {
        return try {
            val logs = blockchainService.getClientAuditLogs(clientId)
            Result.success(logs)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    override suspend fun verifyModelIntegrity(
        modelParams: ModelParameters,
        expectedHash: String
    ): Result<Boolean> {
        return try {
            val actualHash = calculateModelHash(modelParams)
            val isValid = actualHash == expectedHash
            Result.success(isValid)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    override suspend fun checkBlockchainConnection(): Result<Boolean> {
        return try {
            val isConnected = blockchainService.isConnected()
            Result.success(isConnected)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    /**
     * 모델 파라미터의 해시값 계산
     */
    private fun calculateModelHash(modelParams: ModelParameters): String {
        val digest = MessageDigest.getInstance("SHA-256")
        
        // weights를 바이트 배열로 변환
        modelParams.weights.forEach { weightRow ->
            weightRow.forEach { weight ->
                digest.update(weight.toString().toByteArray())
            }
        }
        
        // bias를 바이트 배열로 변환
        modelParams.bias.forEach { bias ->
            digest.update(bias.toString().toByteArray())
        }
        
        // 라운드와 클라이언트 ID도 포함
        digest.update(modelParams.round.toString().toByteArray())
        digest.update(modelParams.clientId.toByteArray())
        
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
