# ==============================================================================
# Foundry Profile: Default (24GB GPUs - e.g. RTX 3090/4090)
# ==============================================================================
# Hermes-4.3-36B (Dense 36B) Q4_K_M (~21.8GB)
#
# A 24GB GPU is the bare minimum for this model. VRAM is extremely tight.
# Context is limited to 8192 to prevent OOM.
# ==============================================================================

PROFILE_CTX_LENGTH=8192         # 8K max context to fit in 24GB
PROFILE_THREADS=8               
PROFILE_THREADS_BATCH=8         
PROFILE_FLASH_ATTN="on"         
PROFILE_KV_TYPE_K="q8_0"        
PROFILE_KV_TYPE_V="q8_0"        
PROFILE_NO_MMAP="true"          
PROFILE_JINJA="true"            
PROFILE_PARALLEL=1              # Strict 1 slot to prevent OOM
PROFILE_PRIO=0                  
PROFILE_CPU_STRICT=0            
PROFILE_CACHE_REUSE=256         
PROFILE_NO_WEBUI="false"        
PROFILE_METRICS="false"         
PROFILE_EXTRA_ARGS=""
