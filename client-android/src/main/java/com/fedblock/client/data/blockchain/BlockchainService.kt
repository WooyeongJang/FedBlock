package com.fedblock.client.data.blockchain

import android.content.Context
import com.fedblock.client.domain.model.AuditLogEntry
import com.fedblock.client.domain.model.AuditEventType
import com.fedblock.client.domain.model.TransactionResult
import org.web3j.abi.FunctionEncoder
import org.web3j.abi.FunctionReturnDecoder
import org.web3j.abi.TypeReference
import org.web3j.abi.datatypes.*
import org.web3j.abi.datatypes.generated.Uint256
import org.web3j.crypto.Credentials
import org.web3j.protocol.Web3j
import org.web3j.protocol.core.DefaultBlockParameterName
import org.web3j.protocol.core.methods.request.Transaction
import org.web3j.protocol.http.HttpService
import org.web3j.tx.RawTransactionManager
import org.web3j.tx.gas.DefaultGasProvider
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.math.BigInteger
import javax.inject.Inject
import javax.inject.Singleton

/**
 * 블록체인 서비스 구현체
 * 
 * FedBlock 스마트 컨트랙트와 상호작용하여
 * 연합학습 감사 로그를 블록체인에 기록합니다.
 */
@Singleton
class BlockchainService @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    // 테스트넷 설정 (실제 배포 시 메인넷으로 변경)
    private val web3j = Web3j.build(HttpService("https://testnet-rpc.monad.xyz"))
    private val contractAddress = "0x7b1d8fB9De56669BA8F38Eba759d53526364774F" // 실제 컨트랙트 주소로 변경
    private val gasProvider = DefaultGasProvider()
    
    // 개발용 프라이빗 키 (실제 배포 시 안전한 키 관리 필요)
    private val privateKey = "0x523d1790742f1749f8bd7c68a41b0e3592f776d9b429f0bb220a0b613a8f4216"
    private val credentials = Credentials.create(privateKey)
    
    /**
     * 연결 상태 확인
     */
    suspend fun isConnected(): Boolean = withContext(Dispatchers.IO) {
        try {
            val clientVersion = web3j.web3ClientVersion().send()
            !clientVersion.hasError()
        } catch (e: Exception) {
            false
        }
    }
    
    /**
     * 감사 이벤트를 블록체인에 기록
     */
    suspend fun logEvent(
        eventType: String,
        clientId: String,
        data: Map<String, String>
    ): TransactionResult = withContext(Dispatchers.IO) {
        try {
            // 데이터를 JSON 문자열로 변환
            val dataJson = data.entries.joinToString(",") { 
                "\"${it.key}\":\"${it.value}\"" 
            }
            val jsonData = "{$dataJson}"
            
            val function = Function(
                "logFederatedLearningEvent",
                listOf(
                    Utf8String(eventType),
                    Utf8String(clientId),
                    Utf8String(jsonData),
                    Uint256(System.currentTimeMillis())
                ) as List<Type<*>>,
                emptyList()
            )
            
            val encodedFunction = FunctionEncoder.encode(function)
            val transactionManager = RawTransactionManager(web3j, credentials)
            
            val transaction = transactionManager.sendTransaction(
                gasProvider.gasPrice,
                gasProvider.gasLimit,
                contractAddress,
                encodedFunction,
                BigInteger.ZERO
            )
            
            TransactionResult(
                hash = transaction.transactionHash,
                success = true,
                gasUsed = transaction.gasUsed,
                blockNumber = transaction.blockNumber
            )
            
        } catch (e: Exception) {
            throw Exception("Failed to log audit event: ${e.message}", e)
        }
    }
    
    /**
     * 클라이언트의 감사 로그 조회
     */
    suspend fun getClientAuditLogs(clientId: String): List<AuditLogEntry> = 
        withContext(Dispatchers.IO) {
            try {
                val function = Function(
                    "getClientEvents",
                    listOf(Utf8String(clientId)) as List<Type<*>>,
                    listOf(
                        object : TypeReference<DynamicArray<Utf8String>>() {},
                        object : TypeReference<DynamicArray<Utf8String>>() {},
                        object : TypeReference<DynamicArray<Utf8String>>() {},
                        object : TypeReference<DynamicArray<Uint256>>() {}
                    )
                )
                
                val encodedFunction = FunctionEncoder.encode(function)
                val ethCall = web3j.ethCall(
                    Transaction.createEthCallTransaction(
                        credentials.address, contractAddress, encodedFunction
                    ),
                    DefaultBlockParameterName.LATEST
                ).send()
                
                if (ethCall.hasError()) {
                    return@withContext emptyList<AuditLogEntry>()
                }
                
                val result = FunctionReturnDecoder.decode(ethCall.value, function.outputParameters)
                if (result.size >= 4) {
                    val eventTypes = (result[0] as DynamicArray<Utf8String>).value
                    val clientIds = (result[1] as DynamicArray<Utf8String>).value
                    val dataList = (result[2] as DynamicArray<Utf8String>).value
                    val timestamps = (result[3] as DynamicArray<Uint256>).value
                    
                    eventTypes.mapIndexed { index, eventType ->
                        AuditLogEntry(
                            id = "$clientId-$index",
                            eventType = AuditEventType.valueOf(eventType.value),
                            clientId = clientIds[index].value,
                            timestamp = timestamps[index].value.toLong(),
                            data = parseJsonData(dataList[index].value)
                        )
                    }
                } else {
                    emptyList()
                }
                
            } catch (e: Exception) {
                emptyList()
            }
        }
    
    /**
     * 모델 파라미터 해시를 블록체인에 기록
     */
    suspend fun recordModelHash(
        clientId: String,
        round: Int,
        modelHash: String
    ): TransactionResult = withContext(Dispatchers.IO) {
        try {
            val function = Function(
                "recordModelHash",
                listOf(
                    Utf8String(clientId),
                    Uint256(BigInteger.valueOf(round.toLong())),
                    Utf8String(modelHash),
                    Uint256(System.currentTimeMillis())
                ) as List<Type<*>>,
                emptyList()
            )
            
            val encodedFunction = FunctionEncoder.encode(function)
            val transactionManager = RawTransactionManager(web3j, credentials)
            
            val transaction = transactionManager.sendTransaction(
                gasProvider.gasPrice,
                gasProvider.gasLimit,
                contractAddress,
                encodedFunction,
                BigInteger.ZERO
            )
            
            TransactionResult(
                hash = transaction.transactionHash,
                success = true,
                gasUsed = transaction.gasUsed,
                blockNumber = transaction.blockNumber
            )
            
        } catch (e: Exception) {
            throw Exception("Failed to record model hash: ${e.message}", e)
        }
    }
    
    /**
     * JSON 데이터 파싱 (간단한 구현)
     */
    private fun parseJsonData(jsonString: String): Map<String, String> {
        return try {
            // 간단한 JSON 파싱 (실제 구현에서는 JSON 라이브러리 사용 권장)
            val cleanJson = jsonString.trim('{', '}')
            if (cleanJson.isEmpty()) {
                emptyMap()
            } else {
                cleanJson.split(",").associate { pair ->
                    val (key, value) = pair.split(":")
                    key.trim('"') to value.trim('"')
                }
            }
        } catch (e: Exception) {
            emptyMap()
        }
    }
}
