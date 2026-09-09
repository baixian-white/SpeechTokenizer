# Variant Manifest

- hand_encoder_base (hand_vs_nas_encoder): changed=encoder architecture: hand-designed; status=completed; notes=single-factor vs nas_encoder_base within debug route
- nas_encoder_base (hand_vs_nas_encoder): changed=encoder architecture: NAS proxy selected encoder; status=completed; notes=single-factor vs hand_encoder_base within debug route
- base_debug (base_vs_lca): changed=Base training only; status=completed; notes=Base vs LCA is diagnostic and changes adaptation objective after Base
- lca_random_l_on (base_vs_lca/random_l_on): changed=LCA objective with random L and ChannelSim; status=completed; notes=diagnostic route
- no_semantic_distill (semantic_distillation_on_off): changed=distill_loss_lambda=0; status=completed; notes=single-factor vs nas_encoder_base except independent short training noise
- random_l_off (random_l_sampling_on_off): changed=random L off; L=3 full-depth only in LCA debug trainer; status=completed; notes=single-factor vs lca_random_l_on within debug route
