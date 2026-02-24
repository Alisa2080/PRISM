#!/usr/bin/env python3
"""
Loss calculators for different visualization scenarios

This module provides loss calculation functions that can be used with 
GenericVITAttentionGradRollout to generate attention visualizations for 
different types of models and tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, Tuple


class SurvivalLossCalculator:
    """
    Loss calculators for survival analysis models
    """
    
    @staticmethod
    def total_risk_sum(risk_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate total risk by summing all time bin risks
        
        Args:
            risk_scores: Tensor of shape [batch_size, n_bins] containing risk scores
        
        Returns:
            Scalar loss for backpropagation
        """
        return risk_scores.sum()
    
    @staticmethod
    def weighted_risk_sum(weights: Optional[torch.Tensor] = None) -> Callable:
        """
        Calculate weighted sum of risks across time bins
        
        Args:
            weights: Optional weights for each time bin [n_bins]
                    If None, uses equal weights
        
        Returns:
            Loss calculation function
        """
        def _calculator(risk_scores: torch.Tensor) -> torch.Tensor:
            if weights is None:
                return risk_scores.mean(dim=1).sum()
            else:
                weighted_scores = risk_scores * weights.to(risk_scores.device)
                return weighted_scores.sum()
        return _calculator
    
    @staticmethod
    def early_time_focused(early_weight: float = 2.0) -> Callable:
        """
        Focus on early time bins (higher weight for earlier time periods)
        
        Args:
            early_weight: Weight multiplier for early time bins
        
        Returns:
            Loss calculation function
        """
        def _calculator(risk_scores: torch.Tensor) -> torch.Tensor:
            n_bins = risk_scores.shape[-1]
            # Create decreasing weights: [early_weight, early_weight*0.8, early_weight*0.6, ...]
            weights = torch.linspace(early_weight, early_weight/n_bins, n_bins).to(risk_scores.device)
            weighted_scores = risk_scores * weights
            return weighted_scores.sum()
        return _calculator
    
    @staticmethod
    def late_time_focused(late_weight: float = 2.0) -> Callable:
        """
        Focus on late time bins (higher weight for later time periods)
        
        Args:
            late_weight: Weight multiplier for late time bins
        
        Returns:
            Loss calculation function
        """
        def _calculator(risk_scores: torch.Tensor) -> torch.Tensor:
            n_bins = risk_scores.shape[-1]
            # Create increasing weights: [late_weight/n_bins, ..., late_weight*0.8, late_weight]
            weights = torch.linspace(late_weight/n_bins, late_weight, n_bins).to(risk_scores.device)
            weighted_scores = risk_scores * weights
            return weighted_scores.sum()
        return _calculator


class ClassificationLossCalculator:
    """
    Loss calculators for classification models
    """
    
    @staticmethod
    def single_class_focus(category_index: int) -> Callable:
        """
        Focus on a specific class for visualization
        
        Args:
            category_index: Index of the class to focus on
        
        Returns:
            Loss calculation function
        """
        def _calculator(logits: torch.Tensor) -> torch.Tensor:
            # Create a mask for the target category
            category_mask = torch.zeros_like(logits)
            category_mask[:, category_index] = 1.0
            loss = (logits * category_mask).sum()
            return loss
        return _calculator
    
    @staticmethod
    def max_probability_class() -> Callable:
        """
        Focus on the class with maximum probability
        
        Returns:
            Loss calculation function
        """
        def _calculator(logits: torch.Tensor) -> torch.Tensor:
            probs = F.softmax(logits, dim=-1)
            max_prob_class = torch.argmax(probs, dim=-1)
            
            # Create mask for max probability class
            category_mask = torch.zeros_like(logits)
            category_mask.scatter_(1, max_prob_class.unsqueeze(1), 1.0)
            loss = (logits * category_mask).sum()
            return loss
        return _calculator
    
    @staticmethod
    def entropy_weighted() -> Callable:
        """
        Weight by entropy (focus on uncertain predictions)
        
        Returns:
            Loss calculation function
        """
        def _calculator(logits: torch.Tensor) -> torch.Tensor:
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            # Use entropy as weight and multiply with max logit
            max_logits = torch.max(logits, dim=-1)[0]
            loss = (entropy * max_logits).sum()
            return loss
        return _calculator


class NLLSurvivalLossCalculator:
    """
    Loss calculators using NLL survival loss for more realistic survival analysis
    
    These calculators use the actual survival loss functions from utils/loss.py
    but require additional survival data (Y, c) which may not be available during
    inference/visualization. They are provided for completeness.
    """
    
    def __init__(self, alpha: float = 0.0):
        """
        Initialize with NLL survival loss parameters
        
        Args:
            alpha: Balance parameter for uncensored vs censored loss
        """
        self.alpha = alpha
    
    def create_synthetic_targets(self, risk_scores: torch.Tensor, 
                               synthetic_strategy: str = 'high_risk_early') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create synthetic survival targets for visualization purposes
        
        Args:
            risk_scores: Risk scores from model [batch_size, n_bins]
            synthetic_strategy: Strategy for creating synthetic targets
                              'high_risk_early': High risk -> early time bin
                              'random': Random targets
        
        Returns:
            Y: Time bin targets [batch_size]
            c: Censorship status [batch_size] (0 = event occurred, 1 = censored)
        """
        batch_size, n_bins = risk_scores.shape
        
        if synthetic_strategy == 'high_risk_early':
            # Higher total risk -> earlier time bin
            total_risk = risk_scores.sum(dim=1)
            # Normalize and convert to time bin (higher risk -> lower bin index)
            normalized_risk = (total_risk - total_risk.min()) / (total_risk.max() - total_risk.min() + 1e-8)
            Y = ((1 - normalized_risk) * (n_bins - 1)).long()
            # Assume all events occurred (not censored)
            c = torch.zeros(batch_size)
        elif synthetic_strategy == 'random':
            Y = torch.randint(0, n_bins, (batch_size,))
            c = torch.bernoulli(torch.full((batch_size,), 0.3))  # 30% censored
        else:
            raise ValueError(f"Unknown synthetic strategy: {synthetic_strategy}")
        
        return Y, c
    
    def nll_based_calculator(self, synthetic_strategy: str = 'high_risk_early') -> Callable:
        """
        Create NLL-based loss calculator with synthetic targets
        
        Args:
            synthetic_strategy: Strategy for creating synthetic survival targets
        
        Returns:
            Loss calculation function
        """
        def _calculator(model_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
            if isinstance(model_output, tuple):
                hazards, S = model_output
            else:
                # Assume output is hazards, compute S
                hazards = torch.sigmoid(model_output)
                S = torch.cumprod(1 - hazards, dim=1)
            
            # Create synthetic targets
            Y, c = self.create_synthetic_targets(hazards, synthetic_strategy)
            Y = Y.to(hazards.device)
            c = c.to(hazards.device)
            
            # Calculate NLL survival loss
            return self._nll_loss(hazards, S, Y, c, self.alpha)
        
        return _calculator
    
    def _nll_loss(self, hazards: torch.Tensor, S: torch.Tensor, 
                  Y: torch.Tensor, c: torch.Tensor, alpha: float = 0.4, eps: float = 1e-7) -> torch.Tensor:
        """
        NLL survival loss implementation (copied from utils/loss.py)
        """
        batch_size = len(Y)
        Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
        c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
        
        if S is None:
            S = torch.cumprod(1 - hazards, dim=1)  # survival is cumulative product of 1 - hazards
        
        # without padding, S(0) = S[0], h(0) = h[0]
        S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
        
        # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
        uncensored_loss = -(1 - c) * (
            torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + 
            torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
        )
        censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        return loss.mean()


# Convenience functions for common use cases
def get_survival_loss_calculator(loss_type: str = 'total_risk', **kwargs) -> Callable:
    """
    Get a survival loss calculator by name
    
    Args:
        loss_type: Type of loss calculator
                  'total_risk' - sum all risk scores
                  'weighted_risk' - weighted sum with custom weights
                  'early_focused' - focus on early time bins
                  'late_focused' - focus on late time bins
                  'nll_synthetic' - NLL loss with synthetic targets
        **kwargs: Additional arguments for specific calculators
    
    Returns:
        Loss calculation function
    """
    if loss_type == 'total_risk':
        return SurvivalLossCalculator.total_risk_sum
    elif loss_type == 'weighted_risk':
        return SurvivalLossCalculator.weighted_risk_sum(kwargs.get('weights', None))
    elif loss_type == 'early_focused':
        return SurvivalLossCalculator.early_time_focused(kwargs.get('early_weight', 2.0))
    elif loss_type == 'late_focused':
        return SurvivalLossCalculator.late_time_focused(kwargs.get('late_weight', 2.0))
    elif loss_type == 'nll_synthetic':
        alpha = kwargs.get('alpha', 0.0)
        strategy = kwargs.get('synthetic_strategy', 'high_risk_early')
        calculator = NLLSurvivalLossCalculator(alpha=alpha)
        return calculator.nll_based_calculator(strategy)
    else:
        raise ValueError(f"Unknown survival loss type: {loss_type}")


def get_classification_loss_calculator(loss_type: str = 'single_class', **kwargs) -> Callable:
    """
    Get a classification loss calculator by name
    
    Args:
        loss_type: Type of loss calculator
                  'single_class' - focus on specific class
                  'max_prob' - focus on max probability class
                  'entropy_weighted' - weight by prediction entropy
        **kwargs: Additional arguments for specific calculators
    
    Returns:
        Loss calculation function
    """
    if loss_type == 'single_class':
        category_index = kwargs.get('category_index', 0)
        return ClassificationLossCalculator.single_class_focus(category_index)
    elif loss_type == 'max_prob':
        return ClassificationLossCalculator.max_probability_class()
    elif loss_type == 'entropy_weighted':
        return ClassificationLossCalculator.entropy_weighted()
    else:
        raise ValueError(f"Unknown classification loss type: {loss_type}")
