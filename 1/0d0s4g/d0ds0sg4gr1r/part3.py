        self.allstep_num = 0
        
        self.zongr = 0
        
        self.bubu = 0

    def init_before_training(self, if_main=True):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| There should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = str(self.gpu_id)
        if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
            self.gpu_id = '0'

        '''set cwd automatically'''
        # if self.cwd is None:
        #     agent_name = self.agent.__class__.__name__
            #self.cwd = f'./{agent_name}/{self.env.env_name}_{self.gpu_id}'
        agent_name = self.agent.__class__.__name__    
        self.cwd = f'./{agent_name}/{self.env.env_name}_{0}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


'''single process training'''




def train_and_evaluate(args,initial,end):
    
    chaochucishu_forshoulian_last = 0
    
    meicitupodebu_0=[]    
    meicitupodebu_1=[]
    meicitupodebu_2=[]
    # meicitupodebu_3=[]
    # meicitupodebu_4=[]
    # meicitupodebu_5=[]
    
    
    

    args.net_dim = int(2 ** 6)
    
    args.max_step = args.env.max_step
    #args.max_memo = (args.max_step - 1) * 8  # capacity of replay buffer
    #args.target_step = args.allstep_num*(21*4)
    
    duoshaogecaiyang = 1
   
    args.target_step = args.allstep_num*(duoshaogecaiyang*2**7) 

    
    args.break_step = 10000000*args.target_step
    
   # print('args.target_step',args.target_step)
    
    args.max_memo =  args.target_step
    
    args.gpu_id = '10'
    
    #args.batch_size = 8 #int(args.max_memo)   # 8 #int(args.max_memo/256)
    
    args.batch_size = 8 # int(args.max_memo)#args.allstep_num*8
    
    #args.batch_size = 32 #int(args.max_memo/256) ##((args.target_step/ args.batch_size)**2)*4
    
    #args.repeat_times = 2**12 # int(2 **6) # repeatedly update network to keep critic's loss small
    
    therefrepeattime = 2**3
    
    
    zongr = 0
    zongbu = 0
    nayibu = 0
    dijicitupo = 0
    
    
    
    for ii in range(duoshaogecaiyang):
        
        exec('r' + str(ii) + '_zuiqiangjiheti' + ' = ' + str(0))
    
    # r1_zuiqiangjiheti = 0
    
    # r2_zuiqiangjiheti = 0

    # r3_zuiqiangjiheti = 0
    
    # r4_zuiqiangjiheti = 0    

    # r5_zuiqiangjiheti = 0
    
    # r6_zuiqiangjiheti = 0

    # r7_zuiqiangjiheti = 0
    
    # r8_zuiqiangjiheti = 0       

    # r9_zuiqiangjiheti = 0
    
    # r10_zuiqiangjiheti = 0

    # r11_zuiqiangjiheti = 0
    
    # r12_zuiqiangjiheti = 0    

    # r13_zuiqiangjiheti = 0
    
    # r14_zuiqiangjiheti = 0

    # r15_zuiqiangjiheti = 0
    
    # r16_zuiqiangjiheti = 0    


    
   # r5_zuiqiangjiheti = 0    
    
    # r3_zuiqiangjiheti = 0
    
    
    # r4_zuiqiangjiheti = 0
    
    
    # log_all_best = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    
    # log_all_best = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    
    # log_all = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    
    log_all_best = [0]
    
    for _ in range(duoshaogecaiyang):
        
        log_all_best.append(0)
    
    log_all_best.append(0)
    
    log_all_best = np.array(log_all_best)
    
    
    log_all = [0]
    
    for _ in range(duoshaogecaiyang):
        
        log_all.append(0)
    
    log_all = np.array(log_all) 
    
    
    
    
    
    recorder_ary = []
    

    
    
    dijicitupo_jihe = []
    
    dianfengdian = []
    
    dianhouzhongdian = []
    
    args.init_before_training()
    


    # geshu =  [100,200,300]
    
    # xuexirate = [1e-4,5e-5,1e-5]
    
    # ratio_clip_beiyong = [0.2,0.2,0.2]
    
    # lambda_entropy_beiyong = [0.02,0.01,0.005]
    
    # repeat_times_zu = [therefrepeattime,therefrepeattime,therefrepeattime]




    geshu =  [200,400,600]
    
    xuexirate = [5e-5,1e-5,1e-6]
    
    ratio_clip_beiyong = [0.3,0.3,0.3]
    
    lambda_entropy_beiyong = [0.01,0.01,0.01]   
    
    repeat_times_zu = [therefrepeattime,therefrepeattime,therefrepeattime]



    # geshu =  [1000,1500,2000]
    
    # xuexirate = [1e-4,1e-5,1e-6]
    
    # ratio_clip_beiyong = [0.15,0.15,0.15]
    
    # lambda_entropy_beiyong = [0.01,0.01,0.01]   
    
    # repeat_times_zu = [2**3,2**3,2**3]
    
    
    xuanxiang = 0
    
    chaobiaozhishiwu = 0
    
    
    
    
    
    
    
    
    jici = 0
    '''basic arguments'''
    cwd = args.cwd
    env = args.env
    agent = args.agent
    gpu_id = args.gpu_id  # necessary for Evaluator?
    bubushu = args.bubushu 

    '''training arguments'''
    net_dim = args.net_dim
    
  #  print('args.max_memo',args.max_memo)
    
    max_memo = args.max_memo
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    if_break_early = args.if_allow_break
    if_per = args.if_per
    gamma = args.gamma
    reward_scale = args.reward_scale
    
    bubu =  args.bubu
    

    '''evaluating arguments'''
    eval_gap = args.eval_gap
    eval_times1 = args.eval_times1
    eval_times2 = args.eval_times2
    if args.env_eval is not None:
        env_eval = args.env_eval
    elif args.env_eval in set(gym.envs.registry.env_specs.keys()):
        env_eval = PreprocessEnv(gym.make(env.env_name))
    else:
        env_eval = deepcopy(env)

    del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: environment'''
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, if_per)
    if_on_policy = getattr(agent, 'if_on_policy', False)

    buffer = ReplayBuffer(max_len=max_memo, state_dim=state_dim, action_dim=1 if if_discrete else action_dim,
                          if_on_policy=if_on_policy, if_per=if_per, if_gpu=True)

    
    
    env_eval_1 = deepcopy(env_eval)
    
    env_eval_2 = deepcopy(env_eval)
    
    for iii in range(duoshaogecaiyang-2):
        
        exec('env_eval_' + str(iii+3) + ' = deepcopy(env_eval)')

    
    
    # env_eval_1 = deepcopy(env_eval)
    
    # env_eval_2 = deepcopy(env_eval)
    
    # env_eval_3 = deepcopy(env_eval)
    
    # env_eval_4 = deepcopy(env_eval)
    
    # env_eval_5 = deepcopy(env_eval)
    
    # env_eval_6 = deepcopy(env_eval)
    
    # env_eval_7 = deepcopy(env_eval)
    
    # env_eval_8 = deepcopy(env_eval)
    
    # env_eval_9 = deepcopy(env_eval)
    
    # env_eval_10 = deepcopy(env_eval)   
    
    # env_eval_11 = deepcopy(env_eval)
    
    # env_eval_12 = deepcopy(env_eval)
    
    # env_eval_13 = deepcopy(env_eval)    
    
    # # env_eval_14 = deepcopy(env_eval)
    
    # # env_eval_15 = deepcopy(env_eval)
    
    # # env_eval_16 = deepcopy(env_eval)
    evaluator1 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_1,
                           eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=1,initial=initial,end=end)
    
    
    # evaluator2 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_2,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=2,initial=initial,end=end)
    
    
    for iii in range(duoshaogecaiyang-2):
        
        fucknnum = iii + 3
        
        exec('evaluator' + str(fucknnum) + ' = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_' + str(fucknnum) + ',eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=' + str(fucknnum) + ',initial=initial,end=end)' )
        
        #print(evaluator1)
    
  
    # evaluator1 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_1,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=1,initial=initial,end=end)
    # evaluator2 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_2,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=2,initial=initial,end=end)
    # evaluator3 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_3,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=3,initial=initial,end=end)
    # evaluator4 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_4,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=4,initial=initial,end=end)

    # evaluator5 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_5,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=5,initial=initial,end=end)
    # evaluator6 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_6,
    #                         eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=6,initial=initial,end=end)
    
    # evaluator7 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_7,
    #                         eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=7,initial=initial,end=end)
    

    # evaluator8 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_8,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=8,initial=initial,end=end)
    # evaluator9 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_9,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=9,initial=initial,end=end)
    # evaluator10 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_10,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=10,initial=initial,end=end)


    # evaluator11 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_11,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=11,initial=initial,end=end)
    # evaluator12 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_12,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=12,initial=initial,end=end)
    # evaluator13 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_13,
    #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=13,initial=initial,end=end)
    # # evaluator14 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_14,
    # #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=14,initial=initial,end=end)

    # # evaluator15 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_15,
    # #                       eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=15,initial=initial,end=end)
    # # evaluator16 = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval_16,
    # #                         eval_gap=eval_gap, eval_times1=eval_times1, eval_times2=eval_times2,bubu=bubu, casenum=16,initial=initial,end=end)




    
    with torch.no_grad():  # 收集
        fuckless = explore_before_training(env, buffer, max_memo , reward_scale, gamma)    
    
    
    '''prepare for training'''
    agent.state = env.reset()
    if if_on_policy:
        steps = 0
    else:  # explore_before_training for off-policy
        with torch.no_grad():  # update replay buffer
            steps = explore_before_training(env, buffer, target_step, reward_scale, gamma)

        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None
    total_step = steps

    '''start training'''

    path_now = evaluator1.cwd
    
    initial_zifu = str(evaluator1.initial)
    
    end_zifu = str(evaluator1.end)      
        
    shenjing_dateguiyi = 'guiyi_' + initial_zifu + 'to' + end_zifu +'_case'+str(evaluator1.casenum)+'_'+ str(evaluator1.bubu)+'bu'+'.pth.npy'
    
    if os.path.exists(shenjing_dateguiyi):
            
        buffer.cunchu = np.load(shenjing_dateguiyi)
        
        buffer.guiyi_initial()




    path_now = evaluator1.cwd

    initial_zifu = str(evaluator1.initial)
    
    end_zifu = str( evaluator1.end )
    
    shenjing = 'actor_' + initial_zifu + 'to' + end_zifu +'_case1_0bu.pth'
    
    if os.path.exists(shenjing_dateguiyi):
    
        agent.act.load_state_dict(torch.load(shenjing))






    path_now = evaluator1.cwd

    initial_zifu = str(evaluator1.initial)
    
    end_zifu = str( evaluator1.end )
    
    shenjing = 'critic_' + initial_zifu + 'to' + end_zifu +'_case1_0bu.pth'
    
    shenjing_tar = 'critic_target_' + initial_zifu + 'to' + end_zifu +'_case1_0bu.pth'
    
    if os.path.exists(shenjing):
        
        agent.cri.load_state_dict(torch.load(shenjing))

        agent.cri_target.load_state_dict(torch.load(shenjing_tar))
        
        print('fuck new cri and cri_target')
    
    
    if_reach_goal = False
    while not ((if_break_early and if_reach_goal)
               or total_step > break_step
               or os.path.exists(f'{cwd}/stop')):


        
        print('————————————————————————————————————————————————————————————————————————————————————————————')
        if xuanxiang == 0:
            
            repeat_times = repeat_times_zu[xuanxiang]
            
            agent.learning_rate = xuexirate[xuanxiang]
            
            agent.optimizer = torch.optim.Adam([{'params': agent.act.parameters(), 'lr': agent.learning_rate},
                                               {'params': agent.cri.parameters(), 'lr': agent.learning_rate}])            
            agent.ratio_clip = ratio_clip_beiyong[xuanxiang]  # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
            agent.lambda_entropy = lambda_entropy_beiyong[xuanxiang]

            
        if zongbu-nayibu == geshu[xuanxiang] :#and zongbu-nayibu >= chaochucishu_forshoulian_last*4 :
            
            dijicitupo_jihe.append(dijicitupo)
            
            dianfengdian.append(nayibu)
            
            dianhouzhongdian.append(zongbu)
            
            print('huanpeizhile')
            
            xuanxiang = xuanxiang + 1
            

            
            
            
            if xuanxiang > len(geshu)-1:
                
            #     xuanxiang = xuanxiang - 1
                
                chaobiaozhishiwu = 1
                
                #buffer.xuejiangeshu(geshu[xuanxiang-1]*40*(2**4))
                
                np.save('over.npy',1)
                
                zidian = {}
                
                zidian['zongbu'] = str(zongbu)
                
                zidian['dangqianjiang'] = str(rping)
                
                zidian['pingjunzuida'] = str(zongr)
                
                zidian['pingjunzuidadebu'] = str(nayibu)
                
                for iii in range(duoshaogecaiyang):
                    
                    fucknnum = iii + 1
                    
                    exec('zidian[\'r'+str(fucknnum) + '\'] = str(r' + str(fucknnum)+')'  )
                    
                    
                    
                for iii in range(duoshaogecaiyang):
                    
                    fucknnum = iii + 1           
                    
                    exec('zidian[\'r'+str(fucknnum) + '_zuiqiangjiheti\'] = str(r' + str(fucknnum)+'_zuiqiangjiheti)'  )
              
                
                # zidian['r1'] = str(r1)
                
                # zidian['r2'] = str(r2)

                # zidian['r3'] = str(r3)
                
                # zidian['r4'] = str(r4)

                # zidian['r5'] = str(r5)
                
                # zidian['r6'] = str(r6)

                # zidian['r7'] = str(r7)
                
                # zidian['r8'] = str(r8)

                # zidian['r9'] = str(r9)
                
                # zidian['r10'] = str(r10)

                # zidian['r11'] = str(r11)
                
                # zidian['r12'] = str(r12)

                # zidian['r13'] = str(r13)

                # zidian['r14'] = str(r14)
                
                # zidian['r15'] = str(r15)

                # zidian['r16'] = str(r16)

                
                # zidian['r1_zuiqiangjiheti'] = str(r1_zuiqiangjiheti)
                
                # zidian['r2_zuiqiangjiheti'] = str(r2_zuiqiangjiheti)

                # zidian['r3_zuiqiangjiheti'] = str(r3_zuiqiangjiheti)
                
                # zidian['r4_zuiqiangjiheti'] = str(r4_zuiqiangjiheti)

                # zidian['r5_zuiqiangjiheti'] = str(r5_zuiqiangjiheti)
                
                # zidian['r6_zuiqiangjiheti'] = str(r6_zuiqiangjiheti)

                # zidian['r7_zuiqiangjiheti'] = str(r7_zuiqiangjiheti)
                
                # zidian['r8_zuiqiangjiheti'] = str(r8_zuiqiangjiheti)

                # zidian['r9_zuiqiangjiheti'] = str(r9_zuiqiangjiheti)
                
                # zidian['r10_zuiqiangjiheti'] = str(r10_zuiqiangjiheti)

                # zidian['r11_zuiqiangjiheti'] = str(r11_zuiqiangjiheti)
                
                # zidian['r12_zuiqiangjiheti'] = str(r12_zuiqiangjiheti)

                # zidian['r13_zuiqiangjiheti'] = str(r13_zuiqiangjiheti)
                
                # # zidian['r14_zuiqiangjiheti'] = str(r14_zuiqiangjiheti)

                # # zidian['r15_zuiqiangjiheti'] = str(r15_zuiqiangjiheti)
                
                # # zidian['r16_zuiqiangjiheti'] = str(r16_zuiqiangjiheti)

                
                zidian['xuanxiang'] = str(xuanxiang)
                
                zidian['dianfengdian'] = str(dianfengdian)
                
                zidian['dijicitupo'] = str(dijicitupo)
                
                zidian['dianfengdianzhongdian'] = str(dianhouzhongdian)
                
                zidian['dijicitupo_jihe'] = str(dijicitupo_jihe)
                
                zidian['meicitupodebu_0'] = str(meicitupodebu_0)
                


                zidian['meicitupodebu_1'] = str(meicitupodebu_1)
                
                zidian['meicitupodebu_2'] = str(meicitupodebu_2)
                
                zidian['chaochucishu_forshoulian_last'] = str(chaochucishu_forshoulian_last)
                
                with open('zongjie.csv', 'w') as f:
                    for key in zidian.keys():
                        f.write("%s,%s\n"%(key,zidian[key])) 

                    f.close()
                
                
                
                
                
                break
            
            nayibu = zongbu
                
            dijicitupo = 0


            # path_now = evaluator1.cwd
            
            # initial_zifu = str(evaluator1.initial)
            
            # end_zifu = str(evaluator1.end)
            
            # shenjing_act = 'actor_' + initial_zifu + 'to' + end_zifu +'_case'+str(evaluator1.casenum)+'_'+ str(evaluator1.bubu)+'bu'+'.pth'
            
            # wangluo_act_path = path_now+ '/'+shenjing_act
            
            # agent.act.load_state_dict(torch.load(wangluo_act_path))
            
            # print('fuck chongxin act')         
            
            
            

            # shenjing_cri = 'critic_' + initial_zifu + 'to' + end_zifu +'_case'+str(evaluator1.casenum)+'_'+ str(evaluator1.bubu)+'bu'+'.pth'
            
            # wangluo_cri_path = path_now+ '/'+shenjing_cri            
            
            # agent.cri.load_state_dict(torch.load(wangluo_cri_path))  

            # print('fuck chongxin cri')            
            
      #      buffer.xuejiangeshu(geshu[xuanxiang-1]*40*(2**4))
 



            # shenjing_cri = 'critic_target_' + initial_zifu + 'to' + end_zifu +'_case'+str(evaluator1.casenum)+'_'+ str(evaluator1.bubu)+'bu'+'.pth'
            
            # wangluo_cri_target_path = path_now+ '/'+shenjing_cri            
            
            # agent.cri_target.load_state_dict(torch.load(wangluo_cri_target_path)) 
            
            # print('fuck chongxin cri_target')


  
                
            # shenjing_dateguiyi = 'guiyi_' + initial_zifu + 'to' + end_zifu +'_case'+str(evaluator1.casenum)+'_'+ str(evaluator1.bubu)+'bu'+'.pth.npy'

            # dataguiyi__path = path_now+ '/'+shenjing_dateguiyi            
            
            # buffer.cunchu = np.load(dataguiyi__path)
            
            # buffer.guiyi_initial()
            
            # print('fuck chongxin guiyi')
            
            repeat_times = repeat_times_zu[xuanxiang]

 
            
            agent.learning_rate = xuexirate[xuanxiang]
            
            agent.optimizer = torch.optim.Adam([{'params': agent.act.parameters(), 'lr': agent.learning_rate},
                                               {'params': agent.cri.parameters(), 'lr': agent.learning_rate}])            
            agent.ratio_clip = ratio_clip_beiyong[xuanxiang]  # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
            agent.lambda_entropy = lambda_entropy_beiyong[xuanxiang]
            
            
            
            
        print('chaochushuliang:  ',zongbu-nayibu)
        
        
        

        with torch.no_grad(): 
        
            time_start = time.time()
            steps = agent.explore_env(env, buffer, target_step, reward_scale, gamma)
            time_end = time.time()
            time_c= time_end - time_start
            print('time cost', time_c, 's')
            
        total_step += steps


        zongbu=zongbu+1
        
        time_start = time.time()

        obj_a, obj_c = agent.update_net(buffer, target_step, batch_size, repeat_times)
        time_end = time.time()
        time_c= time_end - time_start
        print('tratime cost', time_c, 's')
        with torch.no_grad(): 
            
            if_reach_goal1,r1,_,_ = evaluator1.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            evaluator1.draw_plot()   
            
            # if_reach_goal2,r2,_,_ = evaluator2.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator2.draw_plot()
            
            
            for iii in range(duoshaogecaiyang-2):
                
                fucknnum = iii + 3
                
                exec('if_reach_goal'+str(fucknnum) + ',r' + str(fucknnum)+',_,_ = evaluator'+ str(fucknnum) + '.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)'  )
                
                exec('evaluator' + str(fucknnum) + '.draw_plot()' )
                
            # if_reach_goal1,r1,_,_ = evaluator1.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator1.draw_plot()

            # if_reach_goal2,r2,_,_ = evaluator2.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator2.draw_plot()

            # if_reach_goal3,r3,_,_ = evaluator3.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator3.draw_plot()
            
            # if_reach_goal4,r4,_,_ = evaluator4.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator4.draw_plot()        
            
            # if_reach_goal5,r5,_,_ = evaluator5.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator5.draw_plot()  
     
            # if_reach_goal6,r6,_,_ = evaluator6.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator6.draw_plot()         
    
            # if_reach_goal7,r7,_,_ = evaluator7.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator7.draw_plot() 
            
            
            # if_reach_goal8,r8,_,_ = evaluator8.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator8.draw_plot()

            # if_reach_goal9,r9,_,_ = evaluator9.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator9.draw_plot()

            # if_reach_goal10,r10,_,_ = evaluator10.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator10.draw_plot()
     
            
            
            # if_reach_goal11,r11,_,_ = evaluator11.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator11.draw_plot()

            # if_reach_goal12,r12,_,_ = evaluator12.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator12.draw_plot()

            # if_reach_goal13,r13,_,_ = evaluator13.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # evaluator13.draw_plot()
            
            # # if_reach_goal14,r14,_,_ = evaluator14.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # # evaluator14.draw_plot()        
            
            # # if_reach_goal15,r15,_,_ = evaluator15.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # # evaluator15.draw_plot()  
     
            # # if_reach_goal16,r16,_,_ = evaluator16.evaluate_save(agent.act, agent.cri,agent.cri_target, steps, obj_a, obj_c,buffer)
            # # evaluator16.draw_plot()   
        
        rping =    r1
        
        for iii in range(duoshaogecaiyang-1):
            
            fucknnum = iii + 2
            
            xianzhi = locals()['r' + str(fucknnum)]
            
           # print('xianzhi',xianzhi)
            
            rping = rping + xianzhi
            
         #   print('rping',rping)
                  
        
        
        if zongr <= rping:
        
            
        
        
            for iii in range(duoshaogecaiyang):
                
                fucknnum = iii + 1
                
                exec('r' + str(fucknnum) + '_zuiqiangjiheti = r' + str(fucknnum) )
                
                
        
            # r1_zuiqiangjiheti = r1
            
            # r2_zuiqiangjiheti = r2
            
            # r3_zuiqiangjiheti = r3
            
            # r4_zuiqiangjiheti = r4


            # r5_zuiqiangjiheti = r5
            
            # r6_zuiqiangjiheti = r6
            
            # r7_zuiqiangjiheti = r7

            # r8_zuiqiangjiheti = r8
            
            # r9_zuiqiangjiheti = r9
            
            # r10_zuiqiangjiheti = r10
            

            # r11_zuiqiangjiheti = r11
            
            # r12_zuiqiangjiheti = r12
            
            # r13_zuiqiangjiheti = r13
            
            # # r14_zuiqiangjiheti = r14


            # # r15_zuiqiangjiheti = r15
            
            # # r16_zuiqiangjiheti = r16
            

            path_now = os.getcwd()
        
            
            shenjing = 'actor.pth'
            
            path_actor = path_now+ '/AgentPPO/'+shenjing
            
            #act_save_path = f'{self.cwd}/actor.pth'
            torch.save(agent.act.state_dict(), path_actor)
            

            '''save cri.pth'''

            
            shenjing = 'critic.pth'
            
            path_critic = path_now+ '/AgentPPO/'+shenjing
            
            #act_save_path = f'{self.cwd}/actor.pth'
            torch.save(agent.cri.state_dict(), path_critic)                
            
      
            
            shenjing = 'critic_target.pth'
            
            path_critic_tar = path_now+ '/AgentPPO/'+shenjing
            
            #act_save_path = f'{self.cwd}/actor.pth'
            torch.save(agent.cri_target.state_dict(), path_critic_tar)                   
            
            
            
            
            shenjing_dateguiyi = 'guiyi.pth'
            
            dateguiyi_path = path_now+ '/AgentPPO/'+shenjing_dateguiyi
            
            np.save(dateguiyi_path,buffer.cunchu) 


           # print(f"{agent.agent_id:<2}  {zongbu:8.2e}  {rping:8.2f} |")  # save policy and print
                #self.env.render(self.casenum)
                
            # evaluator1.draw_plot(quan = False)
            
            # evaluator2.draw_plot(quan = False)

            # evaluator3.draw_plot(quan = False)
            
            # evaluator4.draw_plot(quan = False)                      

            # evaluator5.draw_plot(quan = False)
            
            # evaluator6.draw_plot(quan = False)   

            # evaluator7.draw_plot(quan = False)                      

            # evaluator8.draw_plot(quan = False)
            
            # evaluator9.draw_plot(quan = False)  
            
            # evaluator10.draw_plot(quan = False)   

            # evaluator11.draw_plot(quan = False)                      

            # evaluator12.draw_plot(quan = False)
            
            # evaluator13.draw_plot(quan = False)            

            zongr = rping
            
            chaochucishu_forshoulian_last = zongbu - nayibu
            
            nayibu = zongbu
            
        #    shangxiachaochu_nayibu = shangxiajiaocuo
            
            
            initial_zifu = str(evaluator1.initial)
            
            end_zifu = str(evaluator1.end)
            
            
            
            needarray=[zongr]

            for iii in range(duoshaogecaiyang):
                
                fucknnum = iii + 1
                
                exec('needarray.append(r' + str(fucknnum) + ')' )
            
            needarray.append(nayibu)

            #log_all_best = np.vstack((log_all_best,np.array([zongr, r1, r2, r3, r4,r5, r6, r7,r8, r9, r10, r11,r12, r13, nayibu])))
            
            
            log_all_best = np.vstack((log_all_best,np.array(needarray)))
            
            #log_all_best = np.vstack((log_all_best,np.array([zongr, r1, r2, nayibu])))
                        
            np.save('bestlog.npy',log_all_best)
            

            dijicitupo = dijicitupo + 1
        
            #evaluator1.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
   
    
            if xuanxiang==0:
                
                meicitupodebu_0.append(nayibu)
                
            if xuanxiang==1:
                
                meicitupodebu_1.append(nayibu)     
                
            if xuanxiang==2:
                
                meicitupodebu_2.append(nayibu) 
            
            # if xuanxiang==3:
                
            #     meicitupodebu_3.append(nayibu)
                
            # if xuanxiang==4:
                
            #     meicitupodebu_4.append(nayibu)     
                
            # if xuanxiang==5:
                
            #     meicitupodebu_5.append(nayibu)
    
            for iii in range(duoshaogecaiyang):
                
                fucknnum = iii + 1
                
                exec('evaluator' + str(fucknnum) + '.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)' )
            
            
            # evaluator1.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator2.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator3.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator4.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)

            # evaluator5.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator6.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)

            # evaluator7.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator8.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)

            # evaluator9.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator10.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator11.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)

            # evaluator12.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator13.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator14.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)

            # evaluator15.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
            
            # evaluator16.chuhaotu(agent.act, steps, obj_a, obj_c,buffer)
        
        needarray_all=[rping]

        for iii in range(duoshaogecaiyang):
            
            fucknnum = iii + 1
            
            exec('needarray_all.append(r' + str(fucknnum) + ')' )
        

        #log_all_best = np.vstack((log_all_best,np.array([zongr, r1, r2, r3, r4,r5, r6, r7,r8, r9, r10, r11,r12, r13, nayibu])))
        
        
        log_all = np.vstack((log_all,np.array(needarray_all)))



       # log_all = np.vstack((log_all,np.array([rping, r1, r2, r3, r4,r5, r6, r7,r8, r9, r10, r11,r12, r13])))
        
        np.save('alllog.npy',log_all)
        
        recorder_ary.append((zongbu, rping, obj_a, obj_c))
        
        reward_learning_curve(recorder_ary)
        
        print('zongbu',zongbu)
        
        print('dangqianjiang',rping)
        
    #    print('dangqianshangxiajiaocuo',shangxiajiaocuo)
        
        print('pingjunzuida',zongr)
        
        print('pingjunzuidadebu',nayibu)
        
     #   print('pingjunzuidashangxiajiaocuo',shangxiachaochu_nayibu)
                
        for iii in range(duoshaogecaiyang):
            
            fucknnum = iii + 1
            
            exec('print(\'' + str(fucknnum) + '  \',r' + str(fucknnum) + ')')
        
        # print('r1  ',r1)
        # print('r2  ',r2)
        # print('r3  ',r3)
        # print('r4  ',r4)
        # print('r5  ',r5)
        # print('r6  ',r6)    
        # print('r7  ',r7)
        # print('r8  ',r8)
        # print('r9  ',r9)
        # print('r10  ',r10)  
        # print('r11  ',r11) 
        # print('r12  ',r12)
        # print('r13  ',r13)
        # print('r14  ',r14)
        # print('r15  ',r15)
        # print('r16  ',r16) 
        for iii in range(duoshaogecaiyang):
            
            fucknnum = iii + 1
            
            exec('print(\'r' + str(fucknnum) + '_zuiqiangjiheti\',r' + str(fucknnum) + '_zuiqiangjiheti)')
            
            
        # print('r1_zuiqiangjiheti', r1_zuiqiangjiheti)
        
        # print('r2_zuiqiangjiheti', r2_zuiqiangjiheti)
        
        # print('r3_zuiqiangjiheti', r3_zuiqiangjiheti)
        
        # print('r4_zuiqiangjiheti', r4_zuiqiangjiheti)

        # print('r5_zuiqiangjiheti', r5_zuiqiangjiheti)
        
        # print('r6_zuiqiangjiheti', r6_zuiqiangjiheti)
        
        # print('r7_zuiqiangjiheti', r7_zuiqiangjiheti)

        # print('r8_zuiqiangjiheti', r8_zuiqiangjiheti)
        
        # print('r9_zuiqiangjiheti', r9_zuiqiangjiheti)
        
        # print('r10_zuiqiangjiheti', r10_zuiqiangjiheti)
                
        # print('r11_zuiqiangjiheti', r11_zuiqiangjiheti)
        
        # print('r12_zuiqiangjiheti', r12_zuiqiangjiheti)
        
        # print('r13_zuiqiangjiheti', r13_zuiqiangjiheti)
        
        # # print('r14_zuiqiangjiheti', r14_zuiqiangjiheti)

        # # print('r15_zuiqiangjiheti', r15_zuiqiangjiheti)
        
        # # print('r16_zuiqiangjiheti', r16_zuiqiangjiheti)
        

        
        print('xuanxiang',xuanxiang)
        
        if chaobiaozhishiwu == 1:
            
            print('chaobiao')
            
        # if neibudiedaidaolema==True and zongbu>3000:
        #     break        
        
        print('dianfengdian',dianfengdian)
        
        print('dijicitupo',dijicitupo)
        
        print('dianfengdianzhongdian',dianhouzhongdian)
        
        print('dijicitupo_jihe',dijicitupo_jihe)
        
        print('meicitupodebu_0',meicitupodebu_0)
        
        print('meicitupodebu_1',meicitupodebu_1)
        
        print('meicitupodebu_2',meicitupodebu_2)
        
        # print('meicitupodebu_3',meicitupodebu_3)
        
        # print('meicitupodebu_4',meicitupodebu_4)
        
        print('chaochucishu_forshoulian_last',chaochucishu_forshoulian_last)
        
        #print('--------------------------------------------------------------------')

        
    #print(f'| SavedDir: {cwd}\n| UsedTime: {time.time() - evaluator.start_time:.0f}')



'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device, casenum,bubu,initial,end):
        self.recorder = [(0., -np.inf, 0., 0., 0.), ]  # total_step, r_avg, r_std, obj_a, obj_c
        self.r_max = -np.inf
        self.total_step = 0

        self.cwd = cwd  # constant
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.env = env
        self.target_return = env.target_return
        
        self.casenum = casenum

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = -1  # a early time
        
        self.bubu = bubu
        
        self.initial = initial
        
        self.end = end
        
        self.neibushu  = 0
        print(f"{'ID':>2}  {'Step':>8}  {'MaxR':>8} |"
              f"{'avgR':>8}  {'stdR':>8}   {'objA':>8}  {'objC':>8} |"
              f"{'avgS':>6}  {'stdS':>4}")

    def evaluate_save(self, act,cri,cri_target, steps, obj_a, obj_c,buffer) -> bool:
        self.total_step += steps  # update total training steps

        if 1:
            self.eval_time = time.time()

            rewards_steps_list = [get_episode_return(self.env, act, self.device,buffer,self.casenum ) for _ in range(self.eval_times1)]
            
            #print('rewards_steps_list',rewards_steps_list)
            
            r_avg, r_std, s_avg, s_std,shangxiajiaocuo = self.get_r_avg_std_s_avg_std(rewards_steps_list)
            
            self.recorder.append((self.total_step, r_avg, r_std, obj_a, obj_c))  # update recorder
                
        #     self.neibushu  = self.neibushu  + 1



        #     if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        #     if if_reach_goal and self.used_time is None:
        #         self.used_time = int(time.time() - self.start_time)
        #         print(f"{'ID':>2}  {'Step':>8}  {'TargetR':>8} |"
        #               f"{'avgR':>8}  {'stdR':>8}   {'UsedTime':>8}  ########\n"
        #               f"{self.agent_id:<2}  {self.total_step:8.2e}  {self.target_return:8.2f} |"
        #               f"{r_avg:8.2f}  {r_std:8.2f}   {self.used_time:>8}  ########")

        #     print(f"{self.casenum:<2}  {self.total_step:8.2e}  {self.r_max:8.2f} |"
        #           f"{r_avg:8.2f}  {r_std:8.2f}   {obj_a:8.2f}  {obj_c:8.2f} |"
        #           f"{s_avg:6.0f}  {s_std:4.0f}")
            
        #     print('----------------------------------------------')
        # else:
        #     if_reach_goal = False
            
        # neibudiedaidaolema = False
        # if self.neibushu > 2000:
        #     neibudiedaidaolema = True
            
        return {},r_avg,{},{}

    def draw_plot(self, quan = True):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None
        
        
        pathh = self.cwd

        # initial = getattr(self.env, 'initial')
        # end = getattr(self.env, 'end')


        initial_zifu = str(self.initial)
        
        end_zifu = str( self.end )


        if quan:
            
            recorderwenjianjia = 'quanrecorder' + initial_zifu + 'to' + end_zifu +'_bu'+ str(self.bubu) +'_case' + str(self.casenum)
            
            if not os.path.exists(pathh+'/' + recorderwenjianjia):
                    
                os.mkdir(pathh+'/' + recorderwenjianjia )
                
            yaowd = pathh+'/' + recorderwenjianjia
            
 
        #data_save = os.path.join(pathh, '/', wenjianjia, '/', str(fucknum)+'.npy')
            data_save =pathh+ '/'+ recorderwenjianjia+ '/'+ 'quan_recorder.npy'
         
            np.save(data_save , self.recorder)
    
            '''draw plot and save as png'''
            train_time = int(time.time() - self.start_time)
            total_step = int(self.recorder[-1][0])
            save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
    
            save_learning_curve(self.recorder, yaowd, save_title)
            
        else:
            
            recorderwenjianjia = 'recorder' + initial_zifu + 'to' + end_zifu +'_bu'+ str(self.bubu) + '_case' + str(self.casenum)
            
            if not os.path.exists(pathh+'/' + recorderwenjianjia):
                    
                os.mkdir(pathh+'/' + recorderwenjianjia )
                
            yaowd = pathh+'/' + recorderwenjianjia
                
            #data_save = os.path.join(pathh, '/', wenjianjia, '/', str(fucknum)+'.npy')
            data_save =pathh+ '/'+ recorderwenjianjia+ '/'+ 'quan_recorder.npy'
             
            np.save(data_save , self.recorder)
        
            '''draw plot and save as png'''
            train_time = int(time.time() - self.start_time)
            total_step = int(self.recorder[-1][0])
            save_title = f"plot_step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"
        
            save_learning_curve(self.recorder, yaowd, save_title)            
            
        
        
        
        

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list)
        r_avg, s_avg,jiaocuo_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std,_ = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std,jiaocuo_avg



    def chuhaotu(self, act, steps, obj_a, obj_c,buffer) -> bool:
        
        get_episode_return_chuhaotu(self.env, act, self.device,buffer,self.casenum )




def get_episode_return(env, act, device,buffer,casenum) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete
    
    
    jiangji_part1 = 0.
    
    jiangji_part2 = 0.
    
    jiangji_part3 = 0.
    
    piancha_1, piancha_2, piancha_3 = 0., 0., 0.

    state = env.reset(casenum)
    for episode_step in range(max_step):
        
        state_out = buffer.guiyi.transform(state.reshape(1,-1))
        
        s_tensor = torch.as_tensor((state_out[0],), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, jiangji = env.step(action)
        episode_return += reward
        
        
        
        jiangji_part1 = jiangji_part1 + jiangji[0]
        
        jiangji_part2 = jiangji_part2 + jiangji[1]
        
        jiangji_part3 = jiangji_part3 + jiangji[2]
        
        piancha_1 = piancha_1 + jiangji[3]
        
        piancha_2 = piancha_2 + jiangji[4]
        
        piancha_3 = piancha_3 + jiangji[5]

        
        
        
        if done:
            break
    
    env.quanrender(casenum, episode_return, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3)
    
    shangxiajiaocuo=0 #= env.jisuanshangxia()
    
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step + 1,shangxiajiaocuo


def save_learning_curve(recorder, cwd='.', save_title='learning curve'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_a = recorder[:, 3]
    obj_c = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.set_xlabel('Total Steps')
    axs0.set_ylabel('Episode Return')
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    axs0.set_xlabel('Total Steps')
    ax11.set_ylabel('objA', color=color11)
    ax11.plot(steps, obj_a, label='objA', color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/plot_learning_curve.png")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()




def reward_learning_curve(recorder):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    obj_a = recorder[:, 2]
    obj_c = recorder[:, 3]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    
    # fig = plt.figure(figsize=(18,9))
    
    # axs = fig.add_subplot(2,1,1)
    
    fig, axs = plt.subplots(2)
    
    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.set_xlabel('Total Steps')
    axs0.set_ylabel('Episode Return')
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    #axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    axs0.set_xlabel('Total Steps')
    ax11.set_ylabel('objA', color=color11)
    ax11.plot(steps, obj_a, label='objA', color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(str(max(r_avg)), y= 2.3 ,  fontsize = 20 )
    plt.savefig('reward_curve.png',dpi = 300)
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()










def explore_before_training(env, buffer, target_step, reward_scale, gamma) -> int:
    # just for off-policy. Because on-policy don't explore before training.
    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    steps = 0
    
    #print('target_step',target_step)
    
    shu = np.empty([target_step,env.state_dim])

    while steps < target_step:
        action = rd.randint(action_dim) if if_discrete else rd.uniform(-1, 1, size=action_dim)
        next_state, reward, done, _ = env.step(action)
        steps += 1

        scaled_reward = reward * reward_scale
        mask = 0.0 if done else gamma
        #other = (scaled_reward, mask, action) if if_discrete else (scaled_reward, mask, *action)
        # buffer.append_buffer(state, (0,0,0,0,0,0))
        shu=np.vstack((shu,next_state))
        
       ## print('shu',shu)

        state = env.reset() if done else next_state
        
    buffer.cunchu = shu
    
    #print('shu',shu)
    
    buffer.guiyi_initial()
    
    return steps

def get_episode_return_chuhaotu(env, act, device,buffer,casenum) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete
    
    
    jiangji_part1 = 0.
    
    jiangji_part2 = 0.
    
    jiangji_part3 = 0.
    
    piancha_1, piancha_2, piancha_3 = 0., 0., 0.


    state = env.reset(casenum)
    for episode_step in range(max_step):
        
        state_out = buffer.guiyi.transform(state.reshape(1,-1))
        
        s_tensor = torch.as_tensor((state_out[0],), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, jiangji = env.step(action)
        episode_return += reward
        
        jiangji_part1 = jiangji_part1 + jiangji[0]
        
        jiangji_part2 = jiangji_part2 + jiangji[1]
        
        jiangji_part3 = jiangji_part2 + jiangji[2]
        
        piancha_1 = piancha_1 + jiangji[3]
        
        piancha_2 = piancha_2 + jiangji[4]
        
        piancha_3 = piancha_3 + jiangji[5]

         
        
        if done:
            break
    
    env.render(casenum, episode_return, jiangji_part1, jiangji_part2, jiangji_part3, piancha_1, piancha_2, piancha_3)
