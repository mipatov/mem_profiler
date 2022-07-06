import torch
import pandas as pd
import time

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

    def one_step_report(self,batch, model, optimizer, do_backward = True,device = torch.device('cpu'),print_loss = False):
    
        report_df = pd.DataFrame(columns=['used_mem','delta_mem','delta_time'])

        delta_time =[0]
        used_mem = [self.gpu_mem()[0]]

        self.gpu_mem_info('begin')

        model.train()
        
        ids = batch['input_ids'].to(device,dtype=torch.long)
        labels = batch['labels'].to(device,dtype=torch.long)
        
        start_time = time.time()

        outputs = model(input_ids = ids,labels = labels)
        loss = outputs[0]

        forward_time = time.time()
        delta_time.append(-start_time + forward_time)

        used_mem.append(self.gpu_mem()[0])
        self.gpu_mem_info(f'{delta_time[-1]:.3f}s forward')
        if do_backward:
            optimizer.zero_grad()
            loss.backward()

            backward_time = time.time()
            delta_time.append(-forward_time + backward_time)
            used_mem.append( self.gpu_mem()[0])
            self.gpu_mem_info(f'{delta_time[-1]:.3f}s backward')

            optimizer.step()

            optimizer_step_time = self.time.time()
            delta_time.append(-backward_time + optimizer_step_time)
            used_mem.append( self.gpu_mem()[0])
            self.gpu_mem_info(f'{delta_time[-1]:.3f}s optimizer_step')
        
        if (print_loss):
            print('loss',loss)

        torch.cuda.empty_cache() 
        used_mem.append( self.gpu_mem()[0])
        end_time = time.time()
        delta_time.append(end_time - optimizer_step_time)
        # 
        report_df.loc[:,'used_mem'] = pd.Series(used_mem)
        report_df.loc[:,'delta_time'] = pd.Series(delta_time)
        indexes = ['begin','forward','backward','optim_step','end']
        report_df.index = indexes

        report_df['delta_mem'] =  report_df['used_mem']- report_df.loc['begin','used_mem']

        report_df.loc['total'] = [self.gpu_mem()[1],0,end_time-start_time]
        return report_df
        