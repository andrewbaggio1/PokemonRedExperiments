# Complete List of Differences: SimpleDQN (pokemon_rl) vs ClusterDQN (cluster_rl)

## 1. Class Name
- **pokemon_rl**: `SimpleDQN`
- **cluster_rl**: `ClusterDQN`

## 2. Imports
- **pokemon_rl**: Basic imports only (`math`, `torch`, `torch.nn`, `torch.nn.functional`)
- **cluster_rl**: Additional type hints (`from __future__ import annotations`, `from typing import Dict, Optional, Tuple`)

## 3. NoisyLinear Class Documentation
- **pokemon_rl**: Has docstring `"""Factorised Gaussian noisy linear layer."""`
- **cluster_rl**: No docstring

## 4. Model Class Docstring
- **pokemon_rl**: 
  ```
  CNN encoder with dual recurrent heads (GRU + LSTM) feeding a dueling
  distributional (quantile) head that uses NoisyNets for exploration.
  ```
- **cluster_rl**: 
  ```
  Unified, efficient model:
  - Conv stem + 2-3 residual blocks (+ optional spatial attention)
  - Global pooling; fuse with compact context MLP
  - Single LSTM for temporal memory
  - Dueling quantile head with NoisyLinear
  - Auxiliary head reconstructs context features
  ```

## 5. __init__ Method Signature
- **pokemon_rl**: 
  ```python
  def __init__(self, obs_shape, map_feat_dim, n_actions, num_quantiles: int = 51):
  ```
  - Positional arguments only
  - Parameter name: `map_feat_dim`
  
- **cluster_rl**: 
  ```python
  def __init__(
      self,
      obs_shape: Tuple[int, int, int],
      context_dim: int,
      n_actions: int,
      *,
      use_spatial_attention: bool = True,
      lstm_hidden_size: int = 512,
      num_quantiles: int = 51,
  ):
  ```
  - Type hints for all parameters
  - Keyword-only arguments after `*` separator
  - Parameter name: `context_dim` (instead of `map_feat_dim`)
  - Additional configurable parameters: `use_spatial_attention`, `lstm_hidden_size`

## 6. Type Conversions in __init__
- **pokemon_rl**: No explicit type conversions
- **cluster_rl**: 
  ```python
  self.n_actions = int(n_actions)
  self.num_quantiles = int(num_quantiles)
  self.lstm_hidden_size = int(lstm_hidden_size)
  ```

## 7. Number of Residual Blocks
- **pokemon_rl**: **3** ResidualBlocks
  ```python
  self.residual = nn.Sequential(
      ResidualBlock(stem_channels, stem_channels),
      ResidualBlock(stem_channels, stem_channels),
      ResidualBlock(stem_channels, stem_channels),
  )
  ```
- **cluster_rl**: **2** ResidualBlocks
  ```python
  self.residual = nn.Sequential(
      ResidualBlock(stem_channels, stem_channels),
      ResidualBlock(stem_channels, stem_channels),
  )
  ```

## 8. Spatial Attention Configuration
- **pokemon_rl**: 
  - Always instantiated (hardcoded)
  - Always applied in forward pass
  ```python
  self.spatial_attn = SpatialAttention(stem_channels, num_heads=4, dropout=0.1)
  # ... in forward:
  features = self.spatial_attn(features)  # Always executed
  ```
  
- **cluster_rl**: 
  - Conditionally instantiated based on `use_spatial_attention` flag
  - Conditionally applied in forward pass
  ```python
  self.use_spatial_attention = bool(use_spatial_attention)
  if self.use_spatial_attention:
      self.spatial_attn = SpatialAttention(stem_channels, num_heads=4, dropout=0.1)
  # ... in forward:
  if self.use_spatial_attention:
      features = self.spatial_attn(features)  # Conditional
  ```

## 9. Context/Map Feature Network Name
- **pokemon_rl**: `self.map_net`
- **cluster_rl**: `self.context_net`
- (Functionally identical, different naming)

## 10. Temporal Memory Architecture
- **pokemon_rl**: **Dual recurrent heads**
  - GRU with hidden size 384
  - LSTM with hidden size 512
  - Outputs concatenated: 384 + 512 = **896 dimensions**
  ```python
  self.gru_hidden_size = 384
  self.lstm_hidden_size = 512
  self.hidden_size = self.gru_hidden_size + self.lstm_hidden_size  # 896
  self.gru = nn.GRU(fused_dim, self.gru_hidden_size, batch_first=True)
  self.lstm = nn.LSTM(fused_dim, self.lstm_hidden_size, batch_first=True)
  ```
  
- **cluster_rl**: **Single LSTM**
  - Only LSTM with configurable hidden size (default 512, config uses 384)
  - Output: **512 dimensions** (or configurable)
  ```python
  self.lstm_hidden_size = int(lstm_hidden_size)  # Default 512, config uses 384
  self.lstm = nn.LSTM(fused_dim, self.lstm_hidden_size, batch_first=True)
  ```

## 11. Hidden State Initialization Return Type
- **pokemon_rl**: Returns **dictionary**
  ```python
  return {"gru": gru_state, "lstm": (lstm_h, lstm_c)}
  ```
  
- **cluster_rl**: Returns **tuple**
  ```python
  return (h0, c0)
  ```

## 12. Hidden State Initialization Implementation
- **pokemon_rl**: 
  - Initializes both GRU and LSTM states
  - GRU: single state tensor
  - LSTM: tuple of (hidden, cell) states
  ```python
  gru_state = torch.zeros(1, batch_size, self.gru_hidden_size, device=device)
  lstm_h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
  lstm_c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
  ```
  
- **cluster_rl**: 
  - Only initializes LSTM states
  ```python
  h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
  c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
  ```

## 13. Forward Method Parameter Names
- **pokemon_rl**: `forward(self, obs, map_feat, hidden=None)`
- **cluster_rl**: `forward(self, obs, context, hidden=None)`

## 14. Forward Method Hidden State Handling
- **pokemon_rl**: 
  - Complex fallback logic for handling different hidden state formats
  - Supports both dict and legacy tuple formats
  - Handles missing keys in dict
  ```python
  if isinstance(hidden, dict):
      gru_hidden = hidden.get("gru")
      lstm_hidden = hidden.get("lstm")
      if gru_hidden is None or lstm_hidden is None:
          init = self.init_hidden(batch, obs.device)
          gru_hidden = init["gru"]
          lstm_hidden = init["lstm"]
  else:
      gru_hidden = hidden
      lstm_hidden = (
          torch.zeros(1, batch, self.lstm_hidden_size, device=obs.device, dtype=obs.dtype),
          torch.zeros(1, batch, self.lstm_hidden_size, device=obs.device, dtype=obs.dtype),
      )
  ```
  
- **cluster_rl**: 
  - Simple tuple handling only
  - No fallback logic needed
  ```python
  # Just uses hidden directly, assumes it's a tuple or None
  ```

## 15. Forward Method Device Transfer Logic
- **pokemon_rl**: 
  - Explicit device transfer for both GRU and LSTM hidden states
  ```python
  gru_hidden = gru_hidden.to(fused.device)
  lstm_hidden = (
      lstm_hidden[0].to(fused.device),
      lstm_hidden[1].to(fused.device),
  )
  ```
  
- **cluster_rl**: 
  - No explicit device transfer (handled by PyTorch automatically or simpler structure)

## 16. Forward Method Recurrent Processing
- **pokemon_rl**: 
  - Processes through both GRU and LSTM
  - Concatenates outputs
  ```python
  gru_output, next_gru = self.gru(fused, gru_hidden)
  lstm_output, (next_lstm_h, next_lstm_c) = self.lstm(fused, lstm_hidden)
  output = torch.cat([gru_output, lstm_output], dim=2).squeeze(1)
  ```
  
- **cluster_rl**: 
  - Processes through single LSTM only
  ```python
  lstm_out, (h1, c1) = self.lstm(fused, hidden)
  output = lstm_out.squeeze(1)
  ```

## 17. Forward Method Next Hidden State Return
- **pokemon_rl**: Returns **dictionary**
  ```python
  next_hidden = {
      "gru": next_gru,
      "lstm": (next_lstm_h, next_lstm_c),
  }
  ```
  
- **cluster_rl**: Returns **tuple**
  ```python
  next_hidden = (h1, c1)
  ```

## 18. Dueling Head Input Dimensions
- **pokemon_rl**: 
  - Input: **896** (GRU 384 + LSTM 512)
  ```python
  NoisyLinear(self.hidden_size, duel_hidden)  # 896 -> 384
  ```
  
- **cluster_rl**: 
  - Input: **512** (or configurable, config uses 384)
  ```python
  NoisyLinear(self.lstm_hidden_size, duel_hidden)  # 512 -> 384 (or 384 -> 384)
  ```

## 19. Auxiliary Head Input Dimensions
- **pokemon_rl**: 
  - Input: **896**
  - Output: `map_feat_dim`
  ```python
  nn.Linear(self.hidden_size, 256)  # 896 -> 256
  nn.Linear(256, map_feat_dim)
  ```
  
- **cluster_rl**: 
  - Input: **512** (or configurable)
  - Output: `context_dim`
  ```python
  nn.Linear(self.lstm_hidden_size, 256)  # 512 -> 256 (or 384 -> 256)
  nn.Linear(256, context_dim)
  ```

## 20. Variable Naming in Forward Pass
- **pokemon_rl**: 
  - `map_embed = self.map_net(map_feat)`
  
- **cluster_rl**: 
  - `ctx = self.context_net(context)`

## 21. Model Complexity Summary
- **pokemon_rl**: 
  - Total hidden dimensions: **896**
  - Parameters: ~43% more than cluster_rl
  - Recurrent operations per step: 2 (GRU + LSTM)
  
- **cluster_rl**: 
  - Total hidden dimensions: **512** (default) or **384** (as configured)
  - Parameters: More efficient
  - Recurrent operations per step: 1 (LSTM only)

## 22. Configurability
- **pokemon_rl**: 
  - Fixed architecture (no runtime configuration)
  - All hyperparameters hardcoded
  
- **cluster_rl**: 
  - Configurable spatial attention (on/off)
  - Configurable LSTM hidden size
  - More flexible for experimentation

## Summary Statistics

| Aspect | pokemon_rl (SimpleDQN) | cluster_rl (ClusterDQN) |
|--------|------------------------|-------------------------|
| Residual Blocks | 3 | 2 |
| Spatial Attention | Always on | Optional (configurable) |
| Recurrent Layers | GRU (384) + LSTM (512) | LSTM (512/384) only |
| Hidden Dimensions | 896 | 512 (default) / 384 (config) |
| Hidden State Format | Dictionary | Tuple |
| Parameter Count | Higher | Lower (~43% fewer) |
| Forward Pass Complexity | Higher (dual recurrent) | Lower (single recurrent) |
| Configurability | Low | High |
| Type Hints | None | Full type hints |

