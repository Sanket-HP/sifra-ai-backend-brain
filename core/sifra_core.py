# core/sifra_core.py

import numpy as np

# -------- HDP-FUSIONNET MODULES --------
from core.hdp_fusionnet.intent import IntentModule
from core.hdp_fusionnet.context import ContextModule
from core.hdp_fusionnet.meaning import MeaningModule
from core.hdp_fusionnet.emotion import EmotionModule

# -------- HDS-UNITY ENGINE MODULES --------
from core.hds_unity.trend_channel import TrendChannel
from core.hds_unity.correlation_channel import CorrelationChannel
from core.hds_unity.variation_channel import VariationChannel
from core.hds_unity.fusion_matrix import FusionMatrix
from core.hds_unity.memory_signature import MemorySignature

# -------- PREPROCESSOR --------
from data.preprocessor import Preprocessor

# -------- SETTINGS --------
from config.settings import Settings

# -------- LOGGER --------
from utils.logger import SifraLogger


class SifraCore:
    """
    Central Brain of SIFRA AI
    Combines:
      - HDP-FusionNet (intent, context, meaning, emotion)
      - HDS-Unity Engine (trend, correlation, variation, fusion, memory)
      - Preprocessing, Logging, Settings
    """

    def __init__(self):
        # Logging
        self.log = SifraLogger("SIFRA_CORE")

        # HDP-FUSIONNET modules
        self.intent = IntentModule()
        self.context = ContextModule()
        self.meaning = MeaningModule()
        self.emotion = EmotionModule()

        # HDS-UNITY modules
        self.trend = TrendChannel()
        self.corr = CorrelationChannel()
        self.variation = VariationChannel()
        self.fusion = FusionMatrix()
        self.memory = MemorySignature()

        # Preprocessor
        self.preprocessor = Preprocessor()

        self.log.info("SIFRA Core initialized successfully.")

    # ------------------------------------------------------
    #  ANALYSIS ONLY (utility for trend option)
    # ------------------------------------------------------
    def analyze_data(self, dataset):
        clean = self.preprocessor.clean(dataset)
        return self.trend.compute_trend(clean)

    # ------------------------------------------------------
    #  FULL REASONING PIPELINE
    # ------------------------------------------------------
    def run(self, goal, dataset):
        """
        Full thinking pipeline used by:
        - analyze
        - predict
        - forecast
        - anomaly
        - insights
        """

        self.log.info(f"Running full pipeline for goal: {goal}")

        # STEP 1 — Preprocess dataset
        clean_data = self.preprocessor.clean(dataset)

        # STEP 2 — HDP: Intent
        intent_vec = self.intent.detect_intent(goal)
        self.log.info(f"Intent Vector: {intent_vec}")

        # STEP 3 — HDP: Context
        context_vec = self.context.detect_context(goal, clean_data)
        self.log.info(f"Context Vector: {context_vec}")

        # STEP 4 — HDP: Meaning = Intent + Context
        meaning_vec = self.meaning.create_meaning(intent_vec, context_vec)
        self.log.info(f"Meaning Vector: {meaning_vec}")

        # STEP 5 — HDP: Emotion (data volatility)
        emotion_score = self.emotion.detect_emotion(clean_data)
        self.log.info(f"Emotion Score: {emotion_score}")

        # STEP 6 — HDS: Trend Channel
        trend_score = self.trend.compute_trend(clean_data)

        # STEP 7 — HDS: Correlation
        corr_score = self.corr.compute_correlation(clean_data)

        # STEP 8 — HDS: Variation
        var_score = self.variation.compute_variation(clean_data)

        # STEP 9 — HDS: Fusion of all pattern channels
        fusion_vector = self.fusion.fuse(trend_score, corr_score, var_score)
        self.log.info(f"Fusion Vector: {fusion_vector}")

        # STEP 10 — HDS: Memory Signature (pattern fingerprint)
        signature = self.memory.generate_signature(fusion_vector)
        self.log.info(f"Memory Signature: {signature}")

        # RETURN FULL INFORMATION
        return {
            "intent_vector": intent_vec,
            "context_vector": context_vec,
            "meaning_vector": meaning_vec,
            "emotion_score": emotion_score,

            "analysis_result": {
                "trend_score": trend_score,
                "correlation_score": corr_score,
                "variation_score": var_score,
                "fusion_vector": fusion_vector.tolist(),
                "memory_signature": signature,
            },

            "message": f"Task '{goal}' executed successfully."
        }

# --- IGNORE ---