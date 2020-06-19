function [FFACE] = readImage(imageHeight, imageWidth, people, withinsample, startFrom)
    FFACE = zeros(people * withinsample, imageHeight * imageWidth);
    % 讀圖
    for k = 1:1:people
        for m = startFrom:2:10
            fileName = ['ORL3232' '\' num2str(k) '\' num2str(m) '.bmp'];
            % imageX 是 32 * 32 的陣列
            imageX = double(imread(fileName));
            % 每一個圖片從 32 * 32 變成 1 * 1024
            % arrange the image into a row vector
            % https://www.mathworks.com/help/matlab/ref/reshape.html
            matchtempF = reshape(imageX, 1, []);
            % MATLAB 求出來的 eigenvector 是直的
            % 所以我們資料用成橫的，為了要相乘
            FFACE((k - 1) * withinsample + (m - startFrom) / 2 + 1, :) = matchtempF;
        end
    end
end