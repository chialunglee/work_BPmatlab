function EBPTA2
    out = [];
    % 讀檔讀進來的資料
    input = readmatrix("iris_in.csv");
    [rowOfInput, colOfInput] = size(input);
    target = readmatrix("iris_out.csv");
    % [rowOfTarget, colOfTarget] = size(target);

    outnet = [];
    % 隱藏層神經元個數
    numberOfHiddenUnit = 46;
    % numberOfOutputUnit = 1;
    % initialize the weight matrix
    % 輸出層神經元的 weight
    % (46,1)
    outputmatrix = zeros(numberOfHiddenUnit, 1);
    for i = 1:1:numberOfHiddenUnit
        for j = 1:1:1
            outputmatrix(i, j)=rand;
        end
    end
    % `colOfInput` 是輸入資料的維度 (4)
    % `hiddenmatrix` 是所有隱藏層神經元的 weight
    hiddenmatrix = zeros(colOfInput, numberOfHiddenUnit);
    for i = 1:1:colOfInput
        for j = 1:1:numberOfHiddenUnit
            hiddenmatrix(i, j)=rand;
        end
    end

    % root mean square error
    RMSE1 = zeros(100, 1);
    RMSE2 = zeros(100, 1);


    % Training
    for epoch = 1:1:100
        % 存訓練的 error
        t1 = [];
        t2 = [];
        % 前面一半的資料當訓練
        for iter = 1:1:rowOfInput
            % 前傳，還沒倒傳遞

            % training
            % (1, 4) * (4, 隱藏層神經元個數) => (1, 隱藏層神經元個數)
            % `hiddensigma` 是 weight 與 data 相乘的總和
            hiddensigma = input(iter,:)*hiddenmatrix;
            % 隱藏層不取 hardlim()
            % 為了非線性的能力
            % sigmoid 扭曲空間
            hiddennet = logsig(hiddensigma);

            % (1, 隱藏層神經元個數) * (隱藏層神經元個數, 輸出層神經元個數) => (1, 1)
            outputsigma = hiddennet*outputmatrix;
            outputnet = purelin(outputsigma);


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
            % delta of outputmatrix ��X�h�� delta
            doutputnet = dpurelin(outputsigma);
            % (目標 - 實際) * transfer 的微分
            error = target(iter) - outputnet;
            deltaoutput = error * doutputnet;
            t1 = [t1;error.^2];
            % 從這以下看不懂
            % 前一層的 delta 傳過來
            tempdelta=deltaoutput*outputmatrix;
            transfer=dlogsig(hiddensigma,logsig(hiddensigma));
            deltahidden=[];
            for i=1:1:numberOfHiddenUnit
                deltahidden=[deltahidden;tempdelta(i)*transfer(i)];
            end
            % 0.025 學習率 aplha 泰勒展開式
            % 注意加號
            newoutputmatrix=outputmatrix+0.025*(deltaoutput*hiddennet)';
            outputmatrix=newoutputmatrix;

            % hidden layer ���üh�v����s
            newhiddenmatrix=hiddenmatrix;
            for i=1:1:numberOfHiddenUnit
                for j=1:1:colOfInput
                    % 有容錯能力
                    % 注意加號
                    newhiddenmatrix(j,i)=hiddenmatrix(j,i)+0.025*deltahidden(i)*input(iter,j);
                end
            end
            hiddenmatrix=newhiddenmatrix;
        end


        RMSE1(epoch) = sqrt(sum(t1)/75);
        RMSE2(epoch) = sqrt(sum(t2)/75);

        fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, sqrt(sum(t1)/75));
    end


    fprintf('\nTotal number of epochs: %g\n', epoch);
    fprintf('Final RMSE: %g\n', RMSE1(epoch));
    figure(1);
    plot(1:epoch,RMSE1(1:epoch),1:epoch,RMSE2(1:epoch));
    legend('Training','Simulation');
    ylabel('RMSE');xlabel('Epoch');



    Train_Correct=0;

    for i=1:75

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

    for i=76:length(input)

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
    plot(76:length(input),target(76:length(input)),76:length(input),outnet(1:75))
    legend('Function','Simulation');
    Train_Percent= (Train_Correct) / 75;
    Simu_Percent= (Simu_Correct) / (length(input)-75);
    Train_correct_percent=Train_Percent
    Simu_correct_percent=Simu_Percent



    figure(3)
    [m,b,r]=postreg(out',target(1:75)');
end