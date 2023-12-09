from collections import OrderedDict
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

class Module(Module):
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        This method has been overridden to prevent models wrapped in a module
        being saved in an unloadable state.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        print("Called overridden _save_to_state_dict")
        print(f'prefix: {prefix}')
        if prefix == 'module.':
            prefix = ''
        for name, param in self._parameters.items():
            if param is not None:
                if 'module.' in name:
                    n = name[7:]
                else:
                    n = name[:]
                destination[prefix + n] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if 'module.' in name:
                n = name[7:]
            else:
                n = name[:]
            if buf is not None and n not in self._non_persistent_buffers_set:
                destination[prefix + n] = buf if keep_vars else buf.detach()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        This method has been overridden to ensure models wrapped in a module can
        be loaded.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        print("Called overridden _load_from_state_dict")
        if prefix == 'module.':
            prefix = ''
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        super()._load_from_state_dict(new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        r"""Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.
        
        This method has been overridden to ensure models wrapped in a module can
        be loaded.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        print("Called overridden load_state_dict")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        return super().load_state_dict(new_state_dict, strict)

class DistributedDataParallel(Module, DistributedDataParallel):
    pass
