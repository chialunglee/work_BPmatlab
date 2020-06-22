function EBPTA2
    imageHeight = 32;
    imageWidth = 32;
    % 有 20 個人
    people = 20;
    % 每個樣本取 5
    withinsample = 5;
    % 1 訓練 2 測試
    FFACE = [readImage(imageHeight, imageWidth, people, withinsample, 1);
             readImage(imageHeight, imageWidth, people, withinsample, 2)];
    out = [];
    % 讀檔讀進來的資料
    input = PCALDA_Train(FFACE);
%     input = (input - mean(input)) ./ (max(input) - min(input));
    [rowOfInput, colOfInput] = size(input);
    target = zeros(2 * people * withinsample, 1);
    for i = 1:1:2
        for j = 1:1:people
            for k = 1:1:withinsample
                target(((i - 1) * (people * withinsample) + (j - 1) * withinsample + k), 1) = j;
            end
        end
    end
%     target = csvread("iris_out.csv");
    % [rowOfTarget, colOfTarget] = size(target);

    outnet = [];
    % 隱藏層神經元個數
    numberOfHiddenUnit = 25;
    numberOfOutputUnit = 1;
    % initialize the weight matrix
    % 輸出層神經元的 weight
    % (46,1)
    outputWeight = zeros(numberOfHiddenUnit, numberOfOutputUnit);
    for i = 1:1:numberOfHiddenUnit
        for j = 1:1:numberOfOutputUnit
            outputWeight(i, j)=rand;
        end
    end
    % `colOfInput` 是輸入資料的維度 (4)
    % `hiddenWeight` 是所有隱藏層神經元的 weight (4, 46)
    hiddenWeight = zeros(colOfInput, numberOfHiddenUnit);
    for i = 1:1:colOfInput
        for j = 1:1:numberOfHiddenUnit
            hiddenWeight(i, j)=rand;
        end
    end
    
    trainCount = 50;

    % root mean square error
    RMSE1 = zeros(trainCount, 1);
    RMSE2 = zeros(trainCount, 1);


    % Training
    for epoch = 1:1:trainCount
        % 存訓練的 error
        t1 = [];
        t2 = [];
        % 前面一半的資料當訓練
        for iter = 1:1:rowOfInput/2
            % 前傳，還沒倒傳遞

            % training
            % (1, 4) * (4, 隱藏層神經元個數) => (1, 隱藏層神經元個數)
            % `hiddensigma` 是 weight 與 data 相乘的總和
            hiddensigma = input(iter,:)*hiddenWeight;
            % 隱藏層不取 hardlim()
            % 為了非線性的能力
            % sigmoid 扭曲空間
            hiddennet = logsig(hiddensigma);

            % (1, 隱藏層神經元個數) * (隱藏層神經元個數, 輸出層神經元個數) => (1, 1)
            outputsigma = hiddennet*outputWeight;
            outputnet = purelin(outputsigma);
            
            doutputnet = dpurelin(outputsigma);
            % (目標 - 實際) * transfer 的微分
            error = target(iter) - outputnet;
            deltaoutput = error * doutputnet;
            t1 = [t1;error.^2];
            % 從這以下看不懂
            % 前一層的 delta 傳過來
            % (1, 1) * (46, 1) 可以
            tempdelta=deltaoutput*outputWeight;
            transfer=dlogsig(hiddensigma, hiddennet);
            deltahidden=[];
            for i=1:1:numberOfHiddenUnit
                deltahidden=[deltahidden;tempdelta(i)*transfer(i)];
            end
            % 0.025 學習率 aplha 泰勒展開式
            % 注意加號
            newoutputWeight=outputWeight+0.025*(deltaoutput*hiddennet)';
            outputWeight=newoutputWeight;

            % hidden layer ���üh�v����s
            newhiddenWeight=hiddenWeight;
            for i=1:1:numberOfHiddenUnit
                for j=1:1:colOfInput
                    % 有容錯能力
                    % 注意加號
                    newhiddenWeight(j,i)=hiddenWeight(j,i)+0.025*deltahidden(i)*input(iter,j);
                end
            end
            hiddenWeight=newhiddenWeight;
        end


        RMSE1(epoch) = sqrt(sum(t1)/(rowOfInput/2));
        RMSE2(epoch) = sqrt(sum(t2)/(rowOfInput/2));

        fprintf('epoch %.0f:  RMSE = %.3f\n',epoch, sqrt(sum(t1)/(rowOfInput/2)));
    end


    fprintf('\nTotal number of epochs: %g\n', epoch);
    fprintf('Final RMSE: %g\n', RMSE1(epoch));
    figure(1);
    plot(1:epoch,RMSE1(1:epoch),1:epoch,RMSE2(1:epoch));
    legend('Training','Simulation');
    ylabel('RMSE');xlabel('Epoch');



    Train_Correct=0;

    for i=1:(rowOfInput/2)

        hiddensigma=input(i,:)*hiddenWeight;
        hiddennet=logsig(hiddensigma);
        outputsigma=hiddennet*outputWeight;
        outputnet=purelin(outputsigma);
        out=[out;outputnet];
            if outputnet > target(i)-0.5 &  outputnet <= target(i)+0.5
                Train_Correct=Train_Correct+ 1;
            end
    end


    Simu_Correct=0;

    for i=((rowOfInput/2) + 1):rowOfInput

        hiddensigma=input(i,:)*hiddenWeight;
        hiddennet=logsig(hiddensigma);
        outputsigma=hiddennet*outputWeight;
        outputnet=purelin(outputsigma);
        outnet=[outnet;outputnet];
            if outputnet > target(i)-0.5 &  outputnet <= target(i)+0.5
                Simu_Correct=Simu_Correct+ 1;
            end
    end
    figure(2);
    plot(((rowOfInput/2) + 1):rowOfInput,target(((rowOfInput/2) + 1):rowOfInput),((rowOfInput/2) + 1):rowOfInput,outnet(1:(rowOfInput/2)))
    legend('Function','Simulation');
    Train_Percent= (Train_Correct) / (rowOfInput/2);
    Simu_Percent= (Simu_Correct) / (rowOfInput-(rowOfInput/2));
    Train_correct_percent=Train_Percent
    Simu_correct_percent=Simu_Percent



    figure(3)
    [m,b,r]=postreg(out',target(1:(rowOfInput/2))');
end