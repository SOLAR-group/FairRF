
# import Chromosome
import random


class MyMutation:
    # def __init__(self):
    #     super().__init__()
        

    def my_mutation(chrom):
        print("My mutation call")
        # print("LENGTH OF POP AT BEGINNING OF MUTATION: ", len(pop))
        # chrom = random.choice(pop) 
        # for model in chrom.model_list:
            # print(model.__dict__.get('name'))
            # if (model is None) or 'lr' in str(model.__dict__.get('name')):
            #     continue
        param_list = chrom.model.__dict__.get('param_ranges')
        rand_list = [random.randint(0,1) for _ in range(len(param_list.keys()))]
        for ind in range(len(rand_list)):
            if rand_list[ind] == 1:
                chrom.model.__dict__.get('hyper_params')[list(param_list)[ind]] = random.choice(param_list[list(param_list)[ind]])
                chrom.model.__dict__.get('ml_model').set_params(**chrom.model.__dict__.get('hyper_params'))
                # model.__dict__.get('ml_model')['is_on'] = True
                chrom.is_changed = True
        chrom.mutation_list = round(random.uniform(0.0, 1.0), 2)
        # print("population after mutation ", new_format_pop)
        return chrom,