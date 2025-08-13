package com.fedblock.client.ui.features.audit

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.fedblock.client.domain.model.AuditLogEntry
import com.fedblock.client.domain.repository.BlockchainAuditRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

/**
 * 감사 로그 화면 ViewModel
 */
@HiltViewModel
class AuditViewModel @Inject constructor(
    private val blockchainAuditRepository: BlockchainAuditRepository
) : ViewModel() {
    
    private val _uiState = MutableStateFlow(AuditUiState())
    val uiState: StateFlow<AuditUiState> = _uiState.asStateFlow()
    
    private val clientId = UUID.randomUUID().toString().take(8)
    
    /**
     * 감사 로그 로드
     */
    fun loadAuditLogs() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, errorMessage = null)
            
            try {
                val result = blockchainAuditRepository.getAuditLogs(clientId)
                
                if (result.isSuccess) {
                    val auditLogs = result.getOrNull() ?: emptyList()
                    _uiState.value = _uiState.value.copy(
                        auditLogs = auditLogs.sortedByDescending { it.timestamp },
                        isLoading = false
                    )
                } else {
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        errorMessage = "Failed to load audit logs: ${result.exceptionOrNull()?.message}"
                    )
                }
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    errorMessage = "Error loading audit logs: ${e.message}"
                )
            }
        }
    }
}

/**
 * 감사 로그 화면 UI 상태
 */
data class AuditUiState(
    val auditLogs: List<AuditLogEntry> = emptyList(),
    val isLoading: Boolean = false,
    val errorMessage: String? = null
)
