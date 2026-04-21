def apply_freeze_policy(model, freeze_modules: list[str]) -> None:
    """
    A small helper so experiments can freeze parts of the model by name.

    Example:
    freeze_modules:
      - beats_encoder
      - convnext_encoder
    """

    for module_name in freeze_modules:
        model.set_module_trainable(module_name, trainable=False)
