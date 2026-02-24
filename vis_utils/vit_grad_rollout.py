import torch
import torch.nn as nn
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def _sort_multiscale_tensors(attentions, gradients, level):
    """
    Intelligent sorting of multi-scale attention and gradient tensors
    
    CRITICAL FIX: Return scale-grouped results instead of flat sorted lists
    This avoids the inconsistency of sorting then cutting by fixed level*scale_idx
    
    This function sorts tensors by scale based on their spatial dimensions:
    - 65x65 or 64x64 -> 20x magnification (scale 0)
    - 17x17 or 16x16 -> 10x magnification (scale 1) 
    - 5x5 or 4x4 -> 5x magnification (scale 2)
    
    Args:
        attentions: List of attention tensors
        gradients: List of gradient tensors  
        level: Number of layers per scale (unused in new logic)
    
    Returns:
        attn_by_scale_dict: Dict[scale_idx -> List[tensors]]
        grad_by_scale_dict: Dict[scale_idx -> List[tensors]]
    """
    def get_scale_from_tensor(tensor):
        """
        Determine scale based on tensor spatial dimensions
        
        ENHANCED: Filter out inter-scale layers by detecting irregular dimensions
        """
        if tensor.dim() >= 3:
            spatial_dim = tensor.shape[-1]  # Last dimension is spatial
            if spatial_dim >= 60:  # 65x65 or 64x64 -> 20x
                return 0
            elif spatial_dim >= 15:  # 17x17 or 16x16 -> 10x  
                return 1
            elif spatial_dim <= 6:  # 5x5 or 4x4 -> 5x
                return 2
            else:
                # Inter-scale layers often have irregular dimensions like 5 tokens
                # Mark as invalid scale to be filtered out
                return -1  # Invalid scale marker
        return 0  # Default to 20x if unclear
    
    print("🔧 Applying intelligent multi-scale sorting with inter-scale filtering...")
    
    # Group tensors by scale
    attn_by_scale = {0: [], 1: [], 2: []}
    grad_by_scale = {0: [], 1: [], 2: []}
    
    # Track filtered tensors
    filtered_attentions = []
    filtered_gradients = []
    
    print(f"Original attention shapes: {[att.shape for att in attentions]}")
    print(f"Original gradient shapes: {[grad.shape for grad in gradients]}")
    
    # Classify attention tensors
    for i, attn in enumerate(attentions):
        scale = get_scale_from_tensor(attn)
        if scale >= 0:  # Valid scale
            attn_by_scale[scale].append(attn)
            print(f"  ✓ Attention {i} ({attn.shape}) -> Scale {scale} ({['20x', '10x', '5x'][scale]})")
        else:
            filtered_attentions.append((i, attn))
            print(f"  ❌ Filtered attention {i} ({attn.shape}) -> Likely inter-scale layer")
    
    # Classify gradient tensors
    for i, grad in enumerate(gradients):
        scale = get_scale_from_tensor(grad)
        if scale >= 0:  # Valid scale
            grad_by_scale[scale].append(grad)
            print(f"  ✓ Gradient {i} ({grad.shape}) -> Scale {scale} ({['20x', '10x', '5x'][scale]})")
        else:
            filtered_gradients.append((i, grad))
            print(f"  ❌ Filtered gradient {i} ({grad.shape}) -> Likely inter-scale layer")
    
    # Validation and statistics
    total_valid_attns = sum(len(attn_by_scale[scale]) for scale in [0, 1, 2])
    total_valid_grads = sum(len(grad_by_scale[scale]) for scale in [0, 1, 2])
    
    print(f"📊 Scale distribution summary:")
    for scale in [0, 1, 2]:
        scale_name = ['20x', '10x', '5x'][scale]
        attn_count = len(attn_by_scale[scale])
        grad_count = len(grad_by_scale[scale])
        print(f"  Scale {scale} ({scale_name}): {attn_count} attentions, {grad_count} gradients")
        
        if attn_count != grad_count:
            print(f"    ⚠️  WARNING: Mismatched counts for scale {scale}")
    
    print(f"  Filtered out: {len(filtered_attentions)} attentions, {len(filtered_gradients)} gradients")
    print(f"  Total valid: {total_valid_attns} attentions, {total_valid_grads} gradients")
    print("🔧 Multi-scale sorting and filtering completed!")
    
    return attn_by_scale, grad_by_scale

def avg_heads(cam, grad=None):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    if grad != None:
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0) # filter negative

    return cam


def grad_rollout(attentions, gradients, discard_ratio,vis_scale='ss',level=3, learnable_weights=False):
    """
    Enhanced grad_rollout with improved dimension validation and error handling
    """
    print(f"grad_rollout called with {len(attentions)} attentions, {len(gradients)} gradients")
    print(f"vis_scale: {vis_scale}, level: {level}, learnable_weights: {learnable_weights}")
    
    # Validate inputs
    if len(attentions) == 0:
        raise ValueError("No attention tensors provided")
    if len(gradients) == 0:
        raise ValueError("No gradient tensors provided")
    
    # Print shapes for debugging
    print("Attention shapes:", [att.shape for att in attentions])
    print("Gradient shapes:", [grad.shape for grad in gradients])
    
    if vis_scale == 'ss':
        # Validate dimensions
        if len(attentions) != len(gradients):
            print(f"WARNING: Attention count ({len(attentions)}) != Gradient count ({len(gradients)})")
        
        result = torch.eye(attentions[0].size(-1))
        # The order of obtaining gradients and attention scores is reversed
        gradients = gradients[::-1]
        
        with torch.no_grad():
            for i, (attention, grad) in enumerate(zip(attentions, gradients)):
                print(f"Processing layer {i}: attention {attention.shape}, grad {grad.shape}")
                
                # Validate tensor dimensions match
                if attention.shape != grad.shape:
                    print(f"ERROR: Dimension mismatch at layer {i}: "
                          f"attention {attention.shape} vs gradient {grad.shape}")
                    raise ValueError(f"Attention and gradient shapes must match at layer {i}")
                
                weights = grad
                attention_heads_fused = (attention*weights).clamp(min=0).mean(axis=1)

                # Drop the lowest attentions, but don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                # FIXED: discard_ratio should remove the LOWEST attention scores, not highest
                # Use largest=False to get the smallest values (lowest attention)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, largest=False)
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                # FIXED: Use keepdim=True to maintain proper broadcasting dimensions
                a = a / a.sum(dim=-1, keepdim=True)
                result = torch.matmul(a, result)
        
        # Look at the total attention between the class token and the image patches
        mask = result[0,0,1:]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        # Safe normalization to avoid NaNs (e.g., when mask is all zeros)
        mask_max = np.max(mask)
        if mask_max > 0:
            mask = mask / mask_max
        else:
            mask = np.zeros_like(mask)
        # Sanitize any potential NaNs/Infs defensively
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Generated single-scale mask: {mask.shape}")
        return mask
    else:
        # Multi-scale processing with enhanced validation
        mask_all = []
        '''
        attentions: [transformer_20x.layer_0,transformer_20x.layer_1,transformer_20x.layer_2,
                    transformer_10x.layer_0,transformer_10x.layer_1,transformer_10x.layer_2,
                    transformer_5x.layer_0,transformer_5x.layer_1,transformer_5x.layer_2]
        '''
        
        # Handle learnable weights extraction
        if learnable_weights:
            if len(attentions) == 0:
                raise ValueError("No attentions available for learnable weights extraction")
            
            w = attentions[-1]
            w = torch.softmax(w, dim=1)
            w = w.detach().numpy()
            print(f"Extracted learnable weights: {w}")
            
            attns = attentions[:-1]
            grads = gradients[1:] if len(gradients) > 1 else []
            
            print(f"After weight extraction: {len(attns)} attentions, {len(grads)} gradients")
        else:
            attns = attentions
            grads = gradients
            w = None
        
        # Validate multi-scale inputs
        expected_total_layers = level * 3  # 3 scales
        if len(attns) != expected_total_layers:
            print(f"WARNING: Expected {expected_total_layers} attention layers for 3 scales, got {len(attns)}")
        if len(grads) != expected_total_layers:
            print(f"WARNING: Expected {expected_total_layers} gradient layers for 3 scales, got {len(grads)}")
        
        # CRITICAL FIX: Restore gradient reversal for proper alignment
        # Gradients captured during backward pass are in reverse order compared to attentions
        # The reversal is REQUIRED to align attention and gradient tensors from same layers
        grads = grads[::-1]  # RESTORED: This is necessary for correct alignment
        
        # ENHANCED FIX: Use intelligent multi-scale grouping
        # Get scale-grouped results instead of flat sorted lists
        attn_by_scale, grad_by_scale = _sort_multiscale_tensors(attns, grads, level)
        
        with torch.no_grad():
            for scale_idx in range(3):
                scale_name = ['20x', '10x', '5x'][scale_idx]
                print(f"Processing scale {scale_idx} ({scale_name})...")
                
                # CRITICAL FIX: Use scale-grouped tensors directly, no more fixed slicing
                attns_curl = attn_by_scale[scale_idx]
                grads_curl = grad_by_scale[scale_idx]
                
                # Handle case where we don't have layers for this scale
                if len(attns_curl) == 0 or len(grads_curl) == 0:
                    print(f"WARNING: No layers found for scale {scale_idx}, skipping")
                    # Create a dummy mask for this scale
                    if scale_idx == 0:  # 20x -> 8x8
                        mask_all.append(np.ones((8, 8)))
                    elif scale_idx == 1:  # 10x -> 4x4
                        mask_all.append(np.ones((4, 4)))
                    else:  # 5x -> 2x2
                        mask_all.append(np.ones((2, 2)))
                    continue
                
                print(f"Scale {scale_idx}: using {len(attns_curl)} attention layers, {len(grads_curl)} gradient layers")
                
                # ENHANCED DEBUGGING: Show attention and gradient alignment for this scale
                print(f"  📊 Attention shapes for scale {scale_idx}: {[att.shape for att in attns_curl]}")
                print(f"  📊 Gradient shapes for scale {scale_idx}: {[grad.shape for grad in grads_curl]}")
                
                # Comprehensive alignment verification
                alignment_valid = True
                for layer_idx, (att, grad) in enumerate(zip(attns_curl, grads_curl)):
                    if att.shape != grad.shape:
                        print(f"  ❌ ERROR: Shape mismatch at scale {scale_idx}, layer {layer_idx}: "
                              f"attention {att.shape} vs gradient {grad.shape}")
                        print(f"      🔍 This indicates incorrect tensor pairing after sorting")
                        alignment_valid = False
                    else:
                        print(f"  ✅ Scale {scale_idx}, layer {layer_idx}: shapes aligned {att.shape}")
                
                if not alignment_valid:
                    print(f"  🚨 Scale {scale_idx} has alignment errors - continuing anyway...")
                else:
                    print(f"  🎯 Scale {scale_idx} alignment validation passed!")
                
                if len(attns_curl) == 0 or len(grads_curl) == 0:
                    print(f"WARNING: No layers available for scale {scale_idx}")
                    continue
                
                # Validate first layer dimensions for this scale
                first_attn = attns_curl[0]
                first_grad = grads_curl[0]
                
                if first_attn.shape != first_grad.shape:
                    print(f"ERROR: Dimension mismatch for scale {scale_idx}: "
                          f"attention {first_attn.shape} vs gradient {first_grad.shape}")
                    raise ValueError(f"Attention and gradient shapes must match for scale {scale_idx}")
                
                result = torch.eye(first_attn.size(-1))
                
                for layer_idx, (attn, grad) in enumerate(zip(attns_curl, grads_curl)):
                    print(f"    🔄 Processing Layer {layer_idx}: attn {attn.shape}, grad {grad.shape}")
                    
                    # Validate dimensions for each layer with detailed error recovery
                    if attn.shape != grad.shape:
                        print(f"    ❌ Layer {layer_idx} dimension mismatch: "
                              f"attention {attn.shape} vs gradient {grad.shape}")
                        print(f"    🔍 DIAGNOSIS: Tensor pairing error after sorting/reversal")
                        print(f"    💡 HINT: This may indicate that the sorting algorithm needs refinement")
                        
                        # Skip this layer-pair but continue processing
                        print(f"    ⚠️  RECOVERY: Skipping layer {layer_idx} at scale {scale_idx}")
                        continue
                    
                    weights = grad
                    attention_heads_fused = (attn * grad).clamp(min=0).mean(axis=1)

                    flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                    # FIXED: discard_ratio should remove the LOWEST attention scores, not highest
                    # Use largest=False to get the smallest values (lowest attention)
                    _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
                    flat[0, indices] = 0
                    
                    I = torch.eye(attention_heads_fused.size(-1))
                    a = (attention_heads_fused + 1.0 * I) / 2
                    # FIXED: Use keepdim=True to maintain proper broadcasting dimensions
                    a = a / a.sum(dim=-1, keepdim=True)
                    result = torch.matmul(a, result)
                
                # Extract final mask for this scale
                mask = result[0, 0, 1:]
                width = int(mask.size(-1) ** 0.5)
                mask = mask.reshape(width, width).numpy()
                # Safe normalization to avoid NaNs (e.g., when mask is all zeros)
                mask_max = np.max(mask)
                if mask_max > 0:
                    mask = mask / mask_max
                else:
                    mask = np.zeros_like(mask)
                # Sanitize any potential NaNs/Infs defensively
                mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
                
                print(f"Generated mask for scale {scale_idx} ({scale_name}): {mask.shape}")
                mask_all.append(mask)
        
        print(f"Generated {len(mask_all)} multi-scale masks")
        
        if learnable_weights:
            return mask_all, w
        return mask_all

def grad_cam(attentions, gradients, vis_scale, level, learnable_weights=False):
    if vis_scale == 'ss':
        print(attentions[-1].shape)
        gradients = gradients[::-1]
        #print(gradients[0].shape)
        with torch.no_grad():
            attn = attentions[-1] # h,s,s
            grad = gradients[-1] # h,s,s

            print(attn.shape)

            
            attn = attn[0,:,0,1:].reshape((-1,int((attn.shape[-1]-1)**0.5),int((attn.shape[-1]-1)**0.5)))
            grad = grad[0,:,0,1:].reshape((-1,int((gradients[-1].shape[-1]-1)**0.5),int((gradients[-1].shape[-1]-1)**0.5)))

            # h,n,n
            cam_grad = (grad*attn).mean(0).clamp(min=0) #n,n
            cam_grad = (cam_grad-cam_grad.min())/(cam_grad.max()-cam_grad.min())
            print(cam_grad)
            

        return cam_grad.numpy()
    else:
        ## multi-scale
        cam_grad_all = []
        if learnable_weights:
            w = attentions[-1]
            w = torch.softmax(w,dim=1)
            w = w.detach().numpy()
            #print(w)
            attns = attentions[:-1]
            grads = gradients[1:]
        else:
            attns = attentions
            grads = gradients
        grads = grads[::-1]
        with torch.no_grad():
            for i in range(3):
                #print(f'attns_len:{len(attns)}')
                attn_mag = attns[level*(i+1)-1]
                grad_mag = grads[level*(i+1)-1]

                print(attn_mag.shape)
                attn = attn_mag[0,:,0,1:].reshape((-1,int((attn_mag.shape[-1]-1)**0.5),int((attn_mag.shape[-1]-1)**0.5)))
                grad = grad_mag[0,:,0,1:].reshape((-1,int((grad_mag.shape[-1]-1)**0.5),int((grad_mag.shape[-1]-1)**0.5)))

                # h,n,n
                cam_grad = (grad*attn).mean(0).clamp(min=0) #n,n
                cam_grad = (cam_grad-cam_grad.min())/(cam_grad.max()-cam_grad.min())
                
                print(cam_grad)
                print(cam_grad.shape)
                cam_grad_all.append(cam_grad.numpy())
        
        if learnable_weights: 
            return cam_grad_all, w
        return cam_grad_all


class GenericVITAttentionGradRollout:
    """
    Generic attention gradient rollout for ViT models with different loss calculation strategies
    
    This unified class supports both classification and survival analysis tasks by accepting
    a custom loss calculation function, making it more flexible and maintainable.
    """
    
    def __init__(self, model, level, 
                 attention_layer_name='attend',
                 discard_ratio=0.9, 
                 vis_type='grad_rollout', 
                 vis_scale='ms', 
                 learnable_weights=False,
                 model_type='survival'):
        """
        Initialize GenericVITAttentionGradRollout
        
        Args:
            model: ViT model (ROAM_Survival_VIS or ROAM_VIS)
            level: Depth of Transformer block
            attention_layer_name: Name pattern for attention layers
            discard_ratio: Proportion of discarded low attention scores
            vis_type: Type of visualization method ('grad_rollout' or 'grad_cam')
            vis_scale: Single scale ('ss') or multi-scale ('ms')
            learnable_weights: Whether weight coefficients are learnable
            model_type: Type of model ('survival' or 'classification')
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.vis_type = vis_type
        self.vis_scale = vis_scale
        self.level = level
        self.learnable_weights = learnable_weights
        self.model_type = model_type
        
        # Register hooks for attention layers with robust checking
        self._register_attention_hooks(attention_layer_name)
        
        self.attentions = []
        self.attention_gradients = []
    
    def _register_attention_hooks(self, attention_layer_name):
        """
        Register attention hooks with improved robustness and validation
        
        Uses both type-based and name-based detection for maximum compatibility
        """
        registered_hooks = 0
        
        # Try type-based registration first (more robust)
        registered_hooks = self._register_hooks_by_type()
        
        # Fallback to name-based registration if type-based fails
        if registered_hooks == 0:
            print("Type-based hook registration failed, falling back to name-based registration...")
            registered_hooks = self._register_hooks_by_name(attention_layer_name)
        
        # Final validation
        expected_hooks = self._get_expected_hook_count()
        if registered_hooks < expected_hooks:
            print(f"WARNING: Hook registration incomplete! "
                  f"Expected {expected_hooks}, registered {registered_hooks}")
            self._print_available_modules()
            
            # Only fail if no hooks were registered at all
            if registered_hooks == 0:
                raise RuntimeError(f"Critical error: No attention hooks were registered! "
                                 f"Please check model structure and layer naming.")
        else:
            print(f"Successfully registered all {registered_hooks} attention hooks")
    
    def _register_hooks_by_type(self):
        """
        Register hooks based on module types (more robust than name matching)
        
        CRITICAL FIX: This function now correctly registers hooks on the 'attend' 
        submodule (nn.Softmax) within Attention/Rel_Attention modules, not on 
        the entire attention module. This ensures we capture attention scores 
        (4D tensor) rather than feature outputs (3D tensor).
        """
        # Import attention classes at runtime to avoid circular imports
        try:
            import sys
            import importlib
            if 'models.ROAM' in sys.modules:
                roam_module = sys.modules['models.ROAM']
            else:
                roam_module = importlib.import_module('models.ROAM')
            
            Attention = getattr(roam_module, 'Attention', None)
            Rel_Attention = getattr(roam_module, 'Rel_Attention', None)
        except:
            print("Could not import attention classes for type-based registration")
            return 0
        
        registered_hooks = 0
        attend_modules = []
        
        # FIXED: Collect 'attend' submodules within attention modules
        for name, module in self.model.named_modules():
            # Check if this is an attention module
            if Attention and isinstance(module, Attention):
                # Find the 'attend' submodule (nn.Softmax) within this Attention module
                if hasattr(module, 'attend'):
                    attend_modules.append((f"{name}.attend", module.attend, 'Attention'))
            elif Rel_Attention and isinstance(module, Rel_Attention):
                # Find the 'attend' submodule (nn.Softmax) within this Rel_Attention module
                if hasattr(module, 'attend'):
                    attend_modules.append((f"{name}.attend", module.attend, 'Rel_Attention'))
        
        print(f"=== Hook Registration ===")
        print(f"Found {len(attend_modules)} attention 'attend' submodules")
        
        # Register hooks based on scale and level requirements
        if self.vis_scale == 'ms':
            # Multi-scale: need attention modules from all 3 scales
            target_count = self.level * 3  # level layers × 3 scales
        else:
            # Single scale: only need transformer_20 attention
            target_count = self.level
        
        # Filter and register appropriate modules
        for name, attend_module, module_type in attend_modules:
            if self._should_register_module(name) and registered_hooks < target_count:
                # FIXED: Register hooks on the 'attend' submodule, not the entire attention module
                attend_module.register_forward_hook(self.get_attention)
                attend_module.register_backward_hook(self.get_attention_gradient)
                print(f"Registered {module_type} hook for: {name}")
                registered_hooks += 1
        
        return registered_hooks
    
    def _register_hooks_by_name(self, attention_layer_name):
        """
        Fallback: Register hooks based on name patterns (legacy approach)
        
        FIXED: Handle both multi-scale and single-scale architecture naming patterns
        """
        registered_hooks = 0
        
        if self.vis_scale == 'ms':
            expected_patterns = [f'vit.transformer_{s}.layers.{l}.0.fn.{attention_layer_name}' 
                               for s in [20, 10, 5] for l in range(self.level)]
        else:
            # FIXED: Single-scale uses 'vit.transformer.layers' not 'vit.transformer_20.layers'
            expected_patterns = [f'vit.transformer.layers.{l}.0.fn.{attention_layer_name}' 
                               for l in range(self.level)]
        
        print(f"Expected attention patterns: {expected_patterns}")
        
        for name, module in self.model.named_modules():
            for pattern in expected_patterns:
                if pattern in name and registered_hooks < len(expected_patterns):
                    # FIXED: Ensure we register on the correct submodule
                    if isinstance(module, nn.Softmax):
                        # This is the attend submodule we want
                        module.register_forward_hook(self.get_attention)
                        module.register_backward_hook(self.get_attention_gradient)
                        print(f"Registered hook for: {name}")
                        registered_hooks += 1
                        break
                    else:
                        # Try to find 'attend' submodule within this module
                        if hasattr(module, 'attend'):
                            module.attend.register_forward_hook(self.get_attention)
                            module.attend.register_backward_hook(self.get_attention_gradient)
                            print(f"Registered hook for: {name}.attend")
                            registered_hooks += 1
                            break
        
        return registered_hooks
    
    def _should_register_module(self, module_name):
        """
        Determine if a module should be registered based on scale and naming
        
        CRITICAL FIX: Handle both multi-scale and single-scale architecture naming
        - Multi-scale: transformer_20.layers, transformer_10.layers, transformer_5.layers
        - Single-scale: transformer.layers (PyramidViT_SingleScale uses different naming)
        - Exclude: transformer_20_to_10, transformer_10_to_5 inter-scale layers
        """
        if self.vis_scale == 'ss':
            # FIXED: Single-scale uses 'transformer.layers' not 'transformer_20.layers'
            # PyramidViT_SingleScale architecture has: vit.transformer.layers.*.0.fn.attend
            return 'transformer.layers' in module_name and 'transformer_20' not in module_name
        else:
            # Multi-scale: only pure intra-scale layers
            # Use .layers to ensure we only match intra-scale transformers
            intra_scale_patterns = [
                'transformer_20.layers',  # Pure 20x layers
                'transformer_10.layers',  # Pure 10x layers  
                'transformer_5.layers'    # Pure 5x layers
            ]
            return any(pattern in module_name for pattern in intra_scale_patterns)
    
    def _get_expected_hook_count(self):
        """
        Calculate expected number of hooks based on configuration
        
        CRITICAL FIX: Handle both multi-scale and single-scale expected hook counts
        - Multi-scale: Only count intra-scale layers (exclude inter-scale)
        - Single-scale: Use depths[0] from PyramidViT_SingleScale, not generic level
        """
        if self.vis_scale == 'ms':
            # For multi-scale: sum of intra-scale layers only
            # Standard ROAM depths=[2,2,2,2,2]: indices [0,2,4] are intra-scale
            if hasattr(self.model, 'vit') and hasattr(self.model.vit, '__class__'):
                model_name = self.model.vit.__class__.__name__
                if model_name == 'PyramidViT_wo_interscale':
                    # No inter-scale layers, all depths are intra-scale
                    return sum(getattr(self.model.vit, 'depths', [self.level] * 3))
                else:
                    # Standard PyramidViT with inter-scale layers
                    # Assume depths=[20x_intra, 20x_to_10x, 10x_intra, 10x_to_5x, 5x_intra]
                    depths = getattr(self.model.vit, 'depths', [self.level] * 5)
                    if len(depths) >= 5:
                        return depths[0] + depths[2] + depths[4]  # Only intra-scale
                    else:
                        return self.level * 3  # Fallback
            else:
                return self.level * 3  # Fallback
        else:
            # FIXED: Single-scale uses PyramidViT_SingleScale with different structure
            # Use actual depths[0] instead of generic level
            if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'transformer'):
                # PyramidViT_SingleScale: transformer has depths[0] layers
                depths = getattr(self.model.vit, 'depths', [self.level] * 5)
                return depths[0]  # Only use the first depth (single transformer block)
            else:
                return self.level  # Fallback
    
    def _print_available_modules(self):
        """Print available modules for debugging"""
        print("Available modules in model:")
        attention_related = []
        for name, module in self.model.named_modules():
            if any(pattern_part in name.lower() for pattern_part in ['transformer', 'attend', 'attention']):
                attention_related.append(f"  - {name} ({type(module).__name__})")
        
        if attention_related:
            for module_info in attention_related:
                print(module_info)
        else:
            print("  No attention-related modules found")
    
    def get_attention(self, module, input, output):
        """
        Hook to capture attention weights
        
        Enhanced with validation and debugging information
        """
        if output is None:
            print("WARNING: Attention hook received None output")
            return
            
        # 验证输出是4D张量（batch, heads, seq_len, seq_len）
        if output.dim() != 4:
            print(f"WARNING: Expected 4D attention tensor, got {output.dim()}D: {output.shape}")
        
        # 移动到CPU并存储
        attention_cpu = output.cpu()
        self.attentions.append(attention_cpu)
        
        # 调试信息
        print(f"Captured attention: {attention_cpu.shape} from {type(module).__name__}")
    
    def get_attention_gradient(self, module, grad_input, grad_output):
        """
        Hook to capture attention gradients
        
        Enhanced with validation and debugging information
        """
        if grad_input is None or len(grad_input) == 0 or grad_input[0] is None:
            print("WARNING: Attention gradient hook received None or empty grad_input")
            return
            
        grad = grad_input[0]
        
        # 验证梯度是4D张量
        if grad.dim() != 4:
            print(f"WARNING: Expected 4D attention gradient, got {grad.dim()}D: {grad.shape}")
        
        # 移动到CPU并存储
        gradient_cpu = grad.cpu()
        self.attention_gradients.append(gradient_cpu)
        
        # 调试信息
        print(f"Captured gradient: {gradient_cpu.shape} from {type(module).__name__}")
    
    def _get_embedding_weights(self):
        """
        Extract embedding weights from the model
        Returns the weights used for combining multi-scale features
        
        IMPROVED: Better handling of different model architectures and clearer debugging
        """
        if not self.learnable_weights:
            # Use fixed weights from model configuration
            if hasattr(self.model, 'embed_weights') and self.model.embed_weights is not None:
                print(f"Using fixed embed_weights from model: {self.model.embed_weights}")
                return self.model.embed_weights
            else:
                # Default equal weights
                default_weights = [1/3, 1/3, 1/3]
                print(f"Using default equal weights: {default_weights}")
                return default_weights
        else:
            # Extract learnable weights from model
            if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'learned_weights'):
                weights = torch.softmax(self.model.vit.learned_weights, dim=0)
                weights_numpy = weights.detach().cpu().numpy().flatten()
                print(f"Using learned weights from model.vit.learned_weights: {weights_numpy}")
                return weights_numpy
            else:
                # FIXED: Removed the problematic ms_attn handling and simplified fallback
                print("Warning: learnable_weights=True but no learnable weights found in model")
                print("Available vit attributes:", [attr for attr in dir(self.model.vit) if not attr.startswith('_')] if hasattr(self.model, 'vit') else "No vit attribute")
                default_weights = [1/3, 1/3, 1/3]
                print(f"Falling back to default weights: {default_weights}")
                return default_weights
    
    def __call__(self, input_tensor, loss_calculator_fn=None):
        """
        Generate attention visualization using custom loss calculation
        
        Args:
            input_tensor: Feature tensor of shape [1, 84, patch_dim]
            loss_calculator_fn: Function to calculate loss from model output
                               If None, uses default based on model_type
        
        Returns:
            masks: List of attention masks for different scales
            weights: Embedding weights
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Clear previous attention captures
        self.attentions = []
        self.attention_gradients = []
        
        # 增强调试信息
        print(f"GenericVITAttentionGradRollout: input_tensor shape = {input_tensor.shape}")
        print(f"Model type: {self.model_type}, vis_scale: {self.vis_scale}, level: {self.level}")
        print(f"Expected attention captures: {self._get_expected_hook_count()}")
        
        # Forward pass with enhanced error handling
        try:
            if self.model_type == 'survival':
                # For survival models - 保持正确的3D输入格式
                if input_tensor.dim() == 3:
                    # [1, 84, dim] 或 [num_ROIs, 84, dim] - 直接使用
                    input_for_model = input_tensor
                    print(f"Using 3D input for survival model: {input_for_model.shape}")
                elif input_tensor.dim() == 2:
                    # [84, dim] -> [1, 84, dim] 添加batch维度
                    input_for_model = input_tensor.unsqueeze(0)
                    print(f"Added batch dimension to input: {input_for_model.shape}")
                else:
                    input_for_model = input_tensor
                    print(f"Using input as-is: {input_for_model.shape}")
                
                output = self.model(input_for_model, vis=True, vis_mode=3)
                if isinstance(output, tuple):
                    risk_scores, _ = output
                else:
                    risk_scores = output
                
                print(f"Model output shape: {risk_scores.shape}")
                
                # Default survival loss: sum of all risk scores
                if loss_calculator_fn is None:
                    loss = risk_scores.sum()
                else:
                    loss = loss_calculator_fn(risk_scores)
                    
                print(f"Calculated loss: {loss.item()}")
                
            else:
                # For classification models
                output = self.model(input_tensor.unsqueeze(0), vis_mode=3)
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                
                # Default classification loss: must provide category_index via loss_calculator_fn
                if loss_calculator_fn is None:
                    raise ValueError("For classification models, loss_calculator_fn must be provided")
                else:
                    loss = loss_calculator_fn(logits)
            
            print(f"Forward pass completed. Captured {len(self.attentions)} attention outputs")
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            print(f"Input tensor shape: {input_tensor.shape}")
            raise
        
        # Backward pass with error handling
        try:
            print("Starting backward pass...")
            loss.backward()
            print(f"Backward pass completed. Captured {len(self.attention_gradients)} gradients")
            
            # 验证注意力捕获的有效性
            if len(self.attentions) == 0:
                raise RuntimeError("No attention weights were captured during forward pass! "
                                 "Check hook registration.")
            if len(self.attention_gradients) == 0:
                raise RuntimeError("No attention gradients were captured during backward pass! "
                                 "Check hook registration and ensure loss.backward() was called.")
            
            # 验证捕获数量的一致性
            expected_count = self._get_expected_hook_count()
            if len(self.attentions) != expected_count:
                print(f"WARNING: Expected {expected_count} attention captures, got {len(self.attentions)}")
            if len(self.attention_gradients) != expected_count:
                print(f"WARNING: Expected {expected_count} gradient captures, got {len(self.attention_gradients)}")
                
        except Exception as e:
            print(f"Error during backward pass: {e}")
            print(f"Captured attentions: {len(self.attentions)}")
            print(f"Captured gradients: {len(self.attention_gradients)}")
            raise
        
        # Get embedding weights
        weights = self._get_embedding_weights()
        print(f"Using embedding weights: {weights}")
        
        # Generate attention visualization using captured gradients
        try:
            print(f"Generating visualization using {self.vis_type}...")
            
            if self.vis_type == 'grad_rollout':
                result = grad_rollout(
                    self.attentions, 
                    self.attention_gradients,
                    self.discard_ratio, 
                    self.vis_scale, 
                    self.level, 
                    self.learnable_weights
                )
            else:
                # grad_cam
                result = grad_cam(
                    self.attentions, 
                    self.attention_gradients, 
                    self.vis_scale, 
                    self.level, 
                    self.learnable_weights
                )
            
            print("Visualization generation completed successfully")
            
        except Exception as e:
            print(f"Error during visualization generation: {e}")
            print(f"Attention shapes: {[att.shape for att in self.attentions]}")
            print(f"Gradient shapes: {[grad.shape for grad in self.attention_gradients]}")
            raise
        
        # Return results in consistent format
        if self.learnable_weights and isinstance(result, tuple):
            masks, _ = result  # Ignore weights from grad_rollout since we extracted them above
            return masks, weights
        else:
            return result, weights


# Legacy wrapper for backward compatibility
class VITSurvivalAttentionGradRollout(GenericVITAttentionGradRollout):
    """
    Backward compatibility wrapper for survival-specific attention gradient rollout
    """
    
    def __init__(self, model, level, 
                 attention_layer_name='attend',
                 discard_ratio=0.9, 
                 vis_type='grad_rollout', 
                 vis_scale='ms', 
                 learnable_weights=False):
        super().__init__(
            model=model,
            level=level,
            attention_layer_name=attention_layer_name,
            discard_ratio=discard_ratio,
            vis_type=vis_type,
            vis_scale=vis_scale,
            learnable_weights=learnable_weights,
            model_type='survival'
        )
    
    def __call__(self, input_tensor):
        """
        Legacy interface for survival analysis
        Uses default survival loss (sum of risk scores)
        """
        return super().__call__(input_tensor, loss_calculator_fn=None)


class VITAttentionGradRollout:
    def __init__(self, model, level, 
                attention_layer_name='attend',
                discard_ratio=0.9, 
                vis_type = 'grad_rollout', 
                vis_scale='ms', 
                learnable_weights=False):
        '''
        ROI-level visualization. generate attention heatmap with self-attention matrix of Transformer
        
        args:
            model: ROAM model for drawing visualization heatmap
            level: depth of Transformer block
            discard_ratio: proportion of discarded low attention scores. focus only on the top attentions
            vis_type: type of visualization method. 'grad_rollout' or 'grad_cam'
                grad_cam: only focus on the last layer of Transformer at each magnification level
                grad_rollou: consider all self-attention layers
            vis_scale" single scale (ss) or multi-scale (ms)
                'ss': only compute heatmap at 20x magnification scale
            learnable_weight: whether weight coefficients of each scale in the model are learnable
                'True': obtain the final weights from the model's state dict
                'False': fixed weight coefficients can be obtained according to initial config
        '''
        self.model = model
        self.discard_ratio = discard_ratio
        self.vis_type = vis_type
        self.vis_scale = vis_scale
        self.level = level
        self.learnable_weights = learnable_weights
        
        if self.vis_scale == 'ms':
            att_layer_name = [f'transformer_{s}.layers.{l}.0.fn.attend' for s in [20,10,5] for l in range(level)]
            if learnable_weights:
                att_layer_name += [f'ms_attn.{level}']

            cur_l = 0
            for name, module in self.model.named_modules():
                if att_layer_name[cur_l] in name:
                    module.register_forward_hook(self.get_attention)
                    module.register_backward_hook(self.get_attention_gradient)

                    cur_l += 1
                    if cur_l >= len(att_layer_name):
                        break

        else:
            # the attention scores of transformer20 are only ones needed
            att_layer_name = [f'transformer_{s}.layers.{l}' for s in [20,10,5] for l in range(level)]

            cur_l = 0
            for name, module in self.model.named_modules():
                if attention_layer_name in name and att_layer_name[cur_l] in name:
                    module.register_forward_hook(self.get_attention)

                    module.register_backward_hook(self.get_attention_gradient)
                    print(name,'is attention')
                    cur_l += 1
                    if cur_l >= level:
                        break
                #print(name)
        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())
    
    # def save_attn_gradients(self, attn_gradients):
    #     print('grad_hook')
    #     print(attn_gradients[0,0,0,1:10])
    #     self.attn_gradients = attn_gradients

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        _,output = self.model(input_tensor.unsqueeze(0),vis_mode=3)
        #print(output.shape)
        loss_fn = nn.CrossEntropyLoss()

        category_mask = torch.zeros(output.size()).cuda()
        category_mask[:, category_index] = 1


        loss = (output*category_mask).sum()

        loss.backward()

        #print(self.vis_type)
        
        if self.vis_type == 'grad_rollout':
            return grad_rollout(self.attentions, self.attention_gradients,
                self.discard_ratio, self.vis_scale,self.level,self.learnable_weights)
        else:
            ## grad_cam
            return grad_cam(self.attentions, self.attention_gradients, self.vis_scale, self.level, self.learnable_weights)
