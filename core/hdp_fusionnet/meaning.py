# core/hdp_fusionnet/meaning.py

class MeaningModule:
    """
    Combines intent vector + context vector into a
    single meaning vector for SIFRA AI.
    """

    def __init__(self):
        print("[HDP-FUSIONNET] Meaning Module Loaded")

    def create_meaning(self, intent_vec, context_vec):
        """
        Meaning vector format:
        [intent..., context...]
        Example:
        [1,0,0,  1,50,3,0.25]
        """

        meaning_vector = intent_vec + context_vec

        return meaning_vector
