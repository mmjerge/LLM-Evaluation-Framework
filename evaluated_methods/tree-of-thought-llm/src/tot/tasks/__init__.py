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
    elif name == "mmlu":
        from  tot.tasks.mmlu import MMLUTask
        return MMLUTask()
    elif name == "aqua":
        from  tot.tasks.aqua import AQUATask
        return AQUATask()
    elif name == "svamp":
        from tot.tasks.svamp import SVAMPTask
        return SVAMPTask()
    else:
        raise NotImplementedError