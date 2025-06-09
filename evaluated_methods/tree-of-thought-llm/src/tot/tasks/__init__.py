def get_task(name):
    if name == 'game24':
        from tot.tasks.game24 import Game24Task
        return Game24Task()
    elif name == 'text':
        from tot.tasks.text import TextTask
        return TextTask()
    elif name == 'crosswords':
        from tot.tasks.crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask()
    elif name == 'gsm8k':
        from tot.tasks.gsm8k import GSM8KTask
        return GSM8KTask()
    elif name == 'gsm-symbolic':
        from tot.tasks.gsm_symbolic import GSMSymbolicTask
        return GSMSymbolicTask()
    elif name == "mmlu":
        from tot.tasks.mmlu import MMLUTask
        return MMLUTask()
    elif name == "aqua":
        from tot.tasks.aqua import AQUATask
        return AQUATask()
    elif name == "svamp":
        from tot.tasks.svamp import SVAMPTask
        return SVAMPTask()
    elif name == "medqa":
        from tot.tasks.medqa import MedQATask
        return MedQATask()
    elif name == "legalbench":
        from tot.tasks.legalbench import LegalBenchTask
        return LegalBenchTask()
    else:
        raise NotImplementedError