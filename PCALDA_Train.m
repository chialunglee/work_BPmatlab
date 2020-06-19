function [pcaTotalFACE] = PCALDA_Train(FFACE)
%     % 有 40 個人
%     people = 20;
%     % 每個樣本取 5
%     withinsample = 5;
    % 32 * 32 == 1024 降維到 50
    principlenum = 50;
%     % 取得圖片尺寸
%     fileName = ['ORL3232' '\' '1' '\' '1' '.bmp'];
%     [imageHeight, imageWidth] = size(imread(fileName));
%     % 訓練資料 startFrom == 1, 測試資料 startFrom == 2
%     % startFrom = 1;
%     % FFACE 變成 200 * 1024
%     FFACE = readImage(imageHeight, imageWidth, people, withinsample, startFrom);
    % 計算 zeroMean
    zeromeanTotalFACE = FFACE - mean(FFACE);

    % pcaSST = cov(zeromeanTotalFACE);
    % "'" Transpose
    % SST == SSB + SSW
    % 全部變異量等於組間變異量加上組內變異量
    pcaCovarianceMatrix = zeromeanTotalFACE' * zeromeanTotalFACE;
    % PCA 是變數名稱紀錄 eigenvector
    % latent 紀錄 eigenvalue
    [pcaEigVector, latent] = eig(pcaCovarianceMatrix);
    % latent 只有對角線有值，其他都是 0
    % 把 latent 存成一維
    eigvalue = diag(latent);
    % 找大的 eigenvalue
    [~, index] = sort(eigvalue, 'descend');
    % 對所有 eigenvector 做排序
    pcaEigVector = pcaEigVector(:, index);

    % projectPCA 1024 * 50
    projectPCA = pcaEigVector(:, 1:principlenum);
    % 200 筆資料全部被降成 50 維
    pcaTotalFACE = zeromeanTotalFACE * projectPCA;
end