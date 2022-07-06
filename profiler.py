import torch
import pandas as pd

class Profiler():

    def __init__(self,) -> None:
        pass

    def gpu_mem(self):
        mem = torch.cuda.mem_get_info()
        mb = list(map(lambda x:x/pow(2,20),mem))
        total = mb[1]
        used = mb[1]-mb[0]
        return used,total

    def gpu_mem_info(self,title = ''):
        used,total = self.gpu_mem()
        print(f'{title} gpu mem : {used:.1f}/{total:.1f} mb')

    def one_step_report(self, batch, model, optimizer, do_backward = True,device = torch.device('cpu')):
        report_df = pd.DataFrame(columns=['used'])
        report_df.loc['begin'] = self.gpu_mem()[0]
        self.gpu_mem_info('begin')
        model.train()
        
        ids = batch['input_ids'].to(device,dtype=torch.long)
        labels = batch['labels'].to(device,dtype=torch.long)
        

        outputs = model(input_ids = ids,labels = labels)
        loss = outputs[0]
        
        report_df.loc['forward'] = self.gpu_mem()[0]
        self.gpu_mem_info('forward')
        if do_backward:
            optimizer.zero_grad()
            loss.backward()
            report_df.loc['backward'] = self.gpu_mem()[0]
            self.gpu_mem_info('backward')
            optimizer.step() 
            report_df.loc['optimizer_step'] = self.gpu_mem()[0]
            self.gpu_mem_info('optimizer_step')
        
        torch.cuda.empty_cache() 
        report_df.loc['end'] = self.gpu_mem()[0]
        report_df['delta'] =  report_df['used']- report_df.loc['begin','used']
        return report_df
        