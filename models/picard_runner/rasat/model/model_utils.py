from .t5_relation_model import T5ForConditionalGeneration as T5_Relation 
from .t5_original_model import T5ForConditionalGeneration as T5_Original
from transformers import T5ForConditionalGeneration as T5_Pretrained


def get_relation_t5_model(config, model_name_or_path):
    if 'checkpoint' in model_name_or_path:
        print("use relation model from checkpoint")
        model = T5_Relation.from_pretrained(model_name_or_path)
    else:
        my_config = config
        model = T5_Relation(config=my_config)

        model_pretrained = T5_Pretrained.from_pretrained(model_name_or_path)
        parameter_dict = model_pretrained.state_dict()
        model_dict = model.state_dict()
        model_dict.update(parameter_dict)
        model.load_state_dict(model_dict)

    return model    


def get_original_t5_model(config, model_name_or_path):
    my_config = config
    model = T5_Original(config=my_config)

    model_pretrained = T5_Pretrained.from_pretrained(model_name_or_path)
    parameter_dict = model_pretrained.state_dict()
    model_dict = model.state_dict()
    model_dict.update(parameter_dict)
    model.load_state_dict(model_dict)

    return model


