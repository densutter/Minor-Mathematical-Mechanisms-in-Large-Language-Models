
from captum.attr import LLMGradientAttribution
from captum.attr._utils.interpretable_input import InterpretableInput
from typing import Union,Optional,Dict,cast
import torch
from torch import Tensor


class LLMGradientAttribution_Features(LLMGradientAttribution):
    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        gen_args: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_length": 25, "do_sample": False}
                    Defaults: None
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance
    
        Returns:
    
            attr (LLMAttributionResult): attribution result
        """
    
        assert isinstance(
            inp, self.SUPPORTED_INPUTS
        ), f"LLMGradAttribution does not support input type {type(inp)}"
    
        if target is None:
            # generate when None
            assert hasattr(self.model, "generate") and callable(self.model.generate), (
                "The model does not have recognizable generate function."
                "Target must be given for attribution"
            )
    
            if not gen_args:
                gen_args = DEFAULT_GEN_ARGS
    
            model_inp = self._format_model_input(inp.to_model_input())
            output_tokens = self.model.generate(model_inp, **gen_args)
            target_tokens = output_tokens[0][model_inp.size(1) :]
        else:
            assert gen_args is None, "gen_args must be None when target is given"
    
            if type(target) is str:
                # exclude sos
                target_tokens = self.tokenizer.encode(target)[1:]
                target_tokens = torch.tensor(target_tokens)
            elif type(target) is torch.Tensor:
                target_tokens = target
    
        attr_inp = inp.to_tensor().to(self.device)
    
        attr_list = []
        for cur_target_idx, _ in enumerate(target_tokens):
            # attr in shape(batch_size, input+output_len, emb_dim)
            attr = self.attr_method.attribute(
                attr_inp,
                additional_forward_args=(
                    inp,
                    target_tokens,
                    cur_target_idx,
                ),
                **kwargs,
            )
            attr = cast(Tensor, attr)
    
            # will have the attr for previous output tokens
            # cut to shape(batch_size, inp_len, emb_dim)
            if cur_target_idx:
                attr = attr[:, :-cur_target_idx]
    
            # the author of IG uses sum
            # https://github.com/ankurtaly/Integrated-Gradients/blob/master/BertModel/bert_model_utils.py#L350
            #print(attr.shape)
            #attr = attr.sum(-1)
            attr_list.append(attr)
    
        # assume inp batch only has one instance
        # to shape(n_output_token, ...)
        
    
        return attr_list
