import numpy as np

BREAK_STR = ";"


class SetTrendingSinusoidPrompts:
    @staticmethod
    def fprompt_base(train_data, train_knowledge, context_data, knowledge, x):
        prompt = "You are an expert in predictiing numerical values based on previously observed data."
        prompt += "Given the following context data point(s) from a new series, you are asked to predict the next y value for the last provided x value."
        prompt += "\n\nContext data:"

        x_context, y_context = context_data

        prompt += f"\n{(BREAK_STR).join([f' {x:.2f}, {y:.2f}' for (x, y) in zip(x_context, y_context)])}{BREAK_STR}"
        prompt += f" {x:.2f}, "

        return prompt

    @staticmethod
    def fprompt_train_data(train_data, train_knowledge, context_data, knowledge, x):
        x_train, y_train = train_data
        x_context, y_context = context_data

        N_train = x_train.shape[0]
        num_targets = x_train.shape[1]

        prompt = "You are an expert in predictiing numerical values based on previously observed data."
        prompt += f"\nYou are given a training dataset of {N_train} example series of data points."
        prompt += "\nEach series is separated by a new line."
        prompt += f'\nEach series is a sequence of {num_targets} data points represented as a list of tuples x, y separated by "{BREAK_STR}".'

        prompt += "\n\nTraining data:"

        for i in range(N_train):
            prompt += f"\n{(BREAK_STR).join([f' {x:.2f}, {y:.2f}' for (x, y) in zip(x_train[i, :, 0], y_train[i, :, 0])])}{BREAK_STR}"

        prompt += "\nGiven the following context data point(s) from a new series, you are asked to predict the next y value for the last provided x value."
        prompt += "\n\nContext data:"

        prompt += f"\n{(BREAK_STR).join([f' {x:.2f}, {y:.2f}' for (x, y) in zip(x_context, y_context)])}{BREAK_STR}"
        prompt += f" {x:.2f}, "

        return prompt

    @staticmethod
    def fprompt_knowledge(train_data, train_knowledge, context_data, knowledge, x):
        x_context, y_context = context_data

        prompt = "You are an expert in predictiing numerical values based on previously observed data and additional knowledge."
        prompt += f"\nKnowledge:\n You know that the data is generated according to y = ax + sin(bx) + c, where {knowledge}."

        if len(x_context) > 0:
            prompt += f'\nYou are also given sample datapoints as a list of tuples x, y separated by "{BREAK_STR}":'
            prompt += "\nYour task is to predict the next y value for the last provided x value."

            prompt += "\nData:"
            prompt += f"\n{(BREAK_STR).join([f' {x:.2f}, {y:.2f}' for (x, y) in zip(x_context, y_context)])}{BREAK_STR}"
            prompt += f" {x:.2f}, "

        else:
            prompt += f"'\n\nIf x={x:.2f}, then the most likely value of y is: y=" ""

        return prompt

    @staticmethod
    def fprompt_train_data_knowledge(
        train_data, train_knowledge, context_data, knowledge, x
    ):
        x_train, y_train = train_data
        x_context, y_context = context_data

        N_train = x_train.shape[0]
        num_targets = x_train.shape[1]

        prompt = "You are an expert in predictiing numerical values based on previously observed data and additional knowledge."
        prompt += f"\nYou are given a training dataset of {N_train} example series of data points and additional knowledge about the data."
        prompt += "\nYou know that each series is generated according to y = ax + sin(bx) + c for some a, b, c."
        prompt += "\nEach series is separated by a new line."
        prompt += f'\nEach series is a sequence of {num_targets} data points represented as a list of tuples x, y separated by "{BREAK_STR}".'
        prompt += "\n\nTraining data:"

        for i in range(N_train):
            prompt += f"\nknowledge: {train_knowledge[i]}  data:'{(BREAK_STR).join([f' {x:.2f}, {y:.2f}' for (x, y) in zip(x_train[i, :, 0], y_train[i, :, 0])])}{BREAK_STR}"

        prompt += "\nGiven the following knowledge and data, you are asked to predict the next y value for the last provided x value."

        prompt += f"\nknowledge: {knowledge}  data:{(BREAK_STR).join([f' {x:.2f}, {y:.2f}' for (x, y) in zip(x_context, y_context)])}{BREAK_STR}"
        prompt += f" {x:.2f}, "

        return prompt

    @staticmethod
    def knowledge_to_str(knowledge):
        k_str = []
        which = np.where(knowledge.sum(axis=0)[:-1] == 1)[0]

        for i in which:
            if i == 0:
                k_str.append(f"a = {knowledge[i, 3]:.2f}")
            elif i == 1:
                k_str.append(f"b = {knowledge[i, 3]:.2f}")
            elif i == 2:
                k_str.append(f"c = {knowledge[i, 3]:.2f}")

        k_str = ", ".join(k_str)
        return k_str
