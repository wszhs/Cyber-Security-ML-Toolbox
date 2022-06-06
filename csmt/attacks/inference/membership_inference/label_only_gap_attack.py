"""
This module implements the Label Only Gap Attack `.
| Paper link: https://arxiv.org/abs/2007.14321
"""
import logging

from csmt.attacks.inference.membership_inference.black_box_rule_based import MembershipInferenceBlackBoxRuleBased


logger = logging.getLogger(__name__)


LabelOnlyGapAttack = MembershipInferenceBlackBoxRuleBased