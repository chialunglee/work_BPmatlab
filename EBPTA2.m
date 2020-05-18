function EBPTA2
input=[];
target=[];
out=[];
s=[];
y=[];
x1=[];
x2=[];
x3=[];
x4=[];


for i=1:1:600
x1=rand;
x2=rand;
x3=rand;
x4=rand;
s=[x1,x2,x3,x4];
input=[input;s];
y=0.8*x1*x2*x3*x4+x1.^2+x2.^2+x3.^3+x4.^2+x1+x2*0.7-x2.^2*x3.^2+0.5*x1*x4.^2+x4*x2.^3+(-x1)*x2+(x1*x2*x3*x4).^3+(x1-x2+x3-x4)+(x1*x4)-(x2*x3)-2;
target=[target;y];
end

outnet=[];

% initialize the weight matrix
outputmatrix=zeros(35,1);
for i=1:1:35
 for j=1:1:1
   outputmatrix(i,j)=rand;
 end
end

hiddenmatrix=zeros(4,35);
for i=1:1:4
 for j=1:1:35
   hiddenmatrix(i,j)=rand;
 end
end


RMSE1=zeros(100,1);
RMSE2=zeros(100,1);


% Training
for epoch=1:1:100
t1=[];
t2=[];
% 前面 400 筆資料當訓練
for iter=1:1:400

% forward �e�ǳ���
% 前傳，還沒倒傳遞

% training
% (1, 4) * (4, 35) => (1, 35)
hiddensigma=input(iter,:)*hiddenmatrix;
% 隱藏層不取 hardlim()
% 為了非線性的能力
% sigmoid 扭曲空間
hiddennet=logsig(hiddensigma);       

% (1, 35) * (35, 1) => (1, 1)
outputsigma=hiddennet*outputmatrix;
outputnet=purelin(outputsigma);    


% simalation 跑 200 筆
% 資料不夠，可以不做
% if iter+400<=600 % take the first 400 as training samples, the remaining 200 as simulations
% hsigma=input(iter+400,:)*hiddenmatrix;
% hnet=logsig(hsigma);       
% 
% osigma=hnet*outputmatrix;
% onet=purelin(osigma);
% 
% mis=target(iter+400)-onet;
% t2=[t2;mis.^2];
% end





% backward part �˶ǳ���
% delta of outputmatrix ��X�h�� delta
doutputnet=dpurelin(outputsigma);
% (目標 - 實際) * transfer 的微分
deltaoutput=(target(iter)-outputnet)*doutputnet;
error=target(iter)-outputnet;
t1=[t1;error.^2];


% delta of hidden layer ���üh�� delta
% 前一層的 delta 傳過來
tempdelta=deltaoutput*outputmatrix;
transfer=dlogsig(hiddensigma,logsig(hiddensigma));
deltahidden=[];
for i=1:1:35
deltahidden=[deltahidden;tempdelta(i)*transfer(i)];
end

% output layer weight update ��X�h�v����s
% 0.025 學習率 aplha 泰勒展開式
newoutputmatrix=outputmatrix+0.025*(deltaoutput*hiddennet)';
outputmatrix=newoutputmatrix;

% hidden layer ���üh�v����s
newhiddenmatrix=hiddenmatrix;
for i=1:1:35
for j=1:1:4
newhiddenmatrix(j,i)=hiddenmatrix(j,i)+0.025*deltahidden(i)*input(iter,j);
end
end
hiddenmatrix=newhiddenmatrix;    
end


RMSE1(epoch) = sqrt(sum(t1)/400);
RMSE2(epoch) = sqrt(sum(t2)/200);

fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, sqrt(sum(t1)/400));
end


fprintf('\nTotal number of epochs: %g\n', epoch);
fprintf('Final RMSE: %g\n', RMSE1(epoch));
figure(1);
plot(1:epoch,RMSE1(1:epoch),1:epoch,RMSE2(1:epoch));
legend('Training','Simulation');
ylabel('RMSE');xlabel('Epoch');



Train_Correct=0;

for i=1:400
    
    hiddensigma=input(i,:)*hiddenmatrix;
    hiddennet=logsig(hiddensigma);       
    outputsigma=hiddennet*outputmatrix;
    outputnet=purelin(outputsigma);
    out=[out;outputnet];
        if outputnet > target(i)-0.5 &  outputnet <= target(i)+0.5
            Train_Correct=Train_Correct+ 1;
        end
end


Simu_Correct=0;

for i=401:length(input)
    
    hiddensigma=input(i,:)*hiddenmatrix;
    hiddennet=logsig(hiddensigma);       
    outputsigma=hiddennet*outputmatrix;
    outputnet=purelin(outputsigma);
    outnet=[outnet;outputnet];
        if outputnet > target(i)-0.5 &  outputnet <= target(i)+0.5
            Simu_Correct=Simu_Correct+ 1;
        end
end
figure(2);
plot(401:length(input),target(401:length(input)),401:length(input),outnet(1:200))
legend('Function','Simulation');
Train_Percent= (Train_Correct) / 400;
Simu_Percent= (Simu_Correct) / (length(input)-400);
Train_correct_percent=Train_Percent
Simu_correct_percent=Simu_Percent



figure(3)
[m,b,r]=postreg(out',target(1:400)');