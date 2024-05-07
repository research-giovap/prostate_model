classdef ensembleModel
    %Copyright 2024, Giovanni Pasini, Department of Mechanical and
    %Aerospace Engineering (DIMA), University of Rome La Sapienza|
    %Institute of Bioimaging and Molecular Physiology, National Research
    %Council (IBFM-CNR)

    %This class represents the ensemble model individuated
    %for addressing the problem of risk stratification prediction in
    %patients affected by prostate cancer.

    properties (Access = private, Constant)
       
        predictors = ["wavelet_LHH_firstorder_Skewness",...
            "wavelet_HHL_glszm_SmallAreaEmphasis",...
            "wavelet_LHL_gldm_DependenceVariance",...
            "wavelet_HHL_glszm_SmallAreaLowGrayLevelEmphasis",...
            "log_sigma_0_5_mm_3D_ngtdm_Busyness",...
            "log_sigma_2_5_mm_3D_glszm_GrayLevelNonUniformity",...
            "wavelet_HHL_gldm_SmallDependenceLowGrayLevelEmphasis",...
            "wavelet_HLH_glszm_GrayLevelNonUniformity",...
            "wavelet_HLH_glcm_Idn",...
            "wavelet_HHL_glcm_Idn",...
            "log_sigma_4_0_mm_3D_firstorder_Skewness"];
      
    end

  
    properties (Access = private)
        dataset
        model

    end

    methods
   
        function [performance, roc, predictions] = evaluateModel(obj,dataset)
            %Use this method to test the model. The test sets should
            %contain the same features used during training.
            %dataset is the ne external test set. Format: table
            
            predCheck = checkPredictors(obj,dataset);
            if predCheck == true
                disp('Continuing with testing....')
                obj.model = load('ensemble_ML_model\ensMlModel.mat');

                y = string(table2array(dataset(:,end)));
                
                
                [dataset1, dataset2, obs] = prepareDataset(obj,dataset);
               
                
                [performance, roc, cm, predictions] = modelPerformance(obj,dataset1,dataset2,y,obs);
                displayResults(obj, performance, roc,predictions)
               
            else
                disp('Error testing aborted, dataset does not contain the predictors used to train the model')
                msgbox('Error testing aborted, dataset does not contain the predictors used to train the model',...
                    'Error',"error")
            end


          
            
        end

        % function classifyObservations(obj,dataset)
        %     predCheck = checkPredictors(obj,dataset);
        %     if predCheck == true
        %         disp('Continuing with classification...')
        %         obj.model = load('ensemble_ML_model\ensMlModel.mat');
        %         dataset = dataset(:,2:end);
        % 
        %         [dataset1, dataset2, obs] = prepareDataset(obj,dataset);
        % 
        % 
        %         [predictions] = modelClassification(obj,dataset1,dataset2,obs);
        %         displayePredictions(obj, predictions)
        % 
        %     else
        %         disp('Error classifying observations aborted, dataset does not contain the predictors used to train the model')
        %         msgbox('Error classifying observations aborted, dataset does not contain the predictors used to train the model',...
        %             'Error',"error")
        %     end
        % end

    end

    methods (Access = private)
        function output = checkPredictors(obj,dataset)
                variableNames = string(dataset.Properties.VariableNames);
                idx = ismember(variableNames,obj.predictors);
                sumIdx = sum(idx);
                if sumIdx == length(obj.predictors)
                    output = true;
                else
                    
                    output = false;

                end
        end
        

        function [output1, output2, obs] = prepareDataset(obj,dataset)
            
            obs = string(table2array(dataset(:,1)));
            dataset = dataset(:,2:end-1);
            output1 = normalize(dataset,'zscore');
            output2 = normalize(dataset,'range');
            
            
        end


        function [performance, rocObjEns, cmEns,predictions] = modelPerformance(obj,testset,testsetO,ytest,obs)
        modelDA1 = obj.model.ensMlModel(1).model;
        modelDA2 = obj.model.ensMlModel(2).model;
        modelDA3 = obj.model.ensMlModel(3).model;
        modelSVM = obj.model.ensMlModel(4).model;
        modelNN = obj.model.ensMlModel(5).model;
        classNames = string(modelDA1.ClassNames);

        [labelDA1_test,ScoreDA1_test] = predict(modelDA1,testset);
        [labelDA2_test,ScoreDA2_test] = predict(modelDA2,testset);
        [labelDA3_test,ScoreDA3_test] = predict(modelDA3,testset);
        labelDAfinal = string(ones(length(labelDA1_test),1));
        idxhigh1 = labelDA1_test == "high risk";
        idxhigh2 = labelDA2_test == "high risk";
        idxhigh3 = labelDA3_test == "high risk";

        idxlow1 = labelDA1_test == "low risk";
        idxlow2 = labelDA2_test == "low risk";
        idxlow3 = labelDA3_test == "low risk";

        votehigh = sum([idxhigh1,idxhigh2,idxhigh3],2);
        votelow = sum([idxlow1,idxlow2,idxlow3],2);

        idxhighmaj = votehigh > votelow;
        labelDAfinal(idxhighmaj) = "high risk";
        labelDAfinal(not(idxhighmaj)) = "low risk";
        labelfinal = string(ones(length(labelDA1_test),1));




        % scoreModelSVM = fitSVMPosterior(modelSVM);
        [labelSVM_test,ScoreSVM_test] = predict(modelSVM,testset);

        rng('twister')
        [labelNN_test,ScoreNN_test] = predict(modelNN,testsetO);
        idxDAh = labelDAfinal == "high risk";
        idxSVMh = labelSVM_test == "high risk";
        idxNNh = labelNN_test == "high risk";

        idxDAl = labelDAfinal == "low risk";
        idxSVMl = labelSVM_test == "low risk";
        idxNNl = labelNN_test == "low risk";

        voteFinalH = sum([idxDAh,idxSVMh,idxNNh],2);
        voteFinalL = sum([idxSVMl,idxDAl,idxNNl],2);
        idxfinalH = voteFinalH>voteFinalL;
        labelfinal(idxfinalH) = "high risk";
        labelfinal(not(idxfinalH)) = "low risk";

        idxacc = ytest == labelfinal;
        predicted_labels = labelfinal;
        accEns = sum(idxacc)/length(idxacc);

        rocObjDA1 = rocmetrics(ytest,ScoreDA1_test,modelDA1.ClassNames);
        auc_DA1 = rocObjDA1.AUC;

        rocObjDA2 = rocmetrics(ytest,ScoreDA2_test,modelDA2.ClassNames);
        auc_DA2 = rocObjDA2.AUC;

        rocObjDA3 = rocmetrics(ytest,ScoreDA3_test,modelDA3.ClassNames);
        auc_DA3 = rocObjDA3.AUC;

        rocObjSVM = rocmetrics(ytest,ScoreSVM_test,modelSVM.ClassNames);
        auc_SVM = rocObjSVM.AUC;

        rocObjNN = rocmetrics(ytest,ScoreNN_test,modelNN.ClassNames);
        auc_NN = rocObjNN.AUC;

        scores_ensemble = (auc_DA1(1)*ScoreDA1_test + auc_DA2(1)*ScoreDA2_test + auc_DA3(1)*ScoreDA3_test + auc_SVM(1)*ScoreSVM_test + auc_NN(1)*ScoreNN_test)/(auc_DA1(1) + auc_DA2(1) + auc_DA3(1) + auc_SVM(1) + auc_NN(1));
        probabilities = scores_ensemble;
        idx_ens_l = predicted_labels == "low risk";
        idx_ens_h = predicted_labels == "high risk";
        probabilities_final = zeros(length(probabilities),1);
        probabilities_final(idx_ens_l) = probabilities(idx_ens_l,2);
        probabilities_final(idx_ens_h) = probabilities(idx_ens_h,1);
        rocObjEns = rocmetrics(ytest,scores_ensemble,modelSVM.ClassNames);
        auc_ens = rocObjEns.AUC;
        cmEns = confusionmat(ytest,labelfinal,"Order",classNames);
        [TPR_ens, TNR_ens, PPV_ens, FScore_ens] = computeCMens(obj, cmEns, classNames);
        performance = [accEns,auc_ens(1),TPR_ens,TNR_ens, PPV_ens, FScore_ens];
        performance = performance([1 2 3 5 7 9]);
        performance = array2table(performance);
        performance.Properties.VariableNames = [
            "accuracy",...
            "auc",...
            "sensitivity",...
            "specificity",...
            "precision",...
            "fscore"
            ];

        predictions = [obs, predicted_labels, probabilities_final];
        predictions = array2table(predictions);
        predictions.Properties.VariableNames = ["observation","risk prediction","probability"];
                    
        end

        function [predictions] = modelClassification(obj,testset,testsetO,ytest,obs)
            modelDA1 = obj.model.ensMlModel(1).model;
            modelDA2 = obj.model.ensMlModel(2).model;
            modelDA3 = obj.model.ensMlModel(3).model;
            modelSVM = obj.model.ensMlModel(4).model;
            modelNN = obj.model.ensMlModel(5).model;
            

            [labelDA1_test,ScoreDA1_test] = predict(modelDA1,testset);
            [labelDA2_test,ScoreDA2_test] = predict(modelDA2,testset);
            [labelDA3_test,ScoreDA3_test] = predict(modelDA3,testset);
            labelDAfinal = string(ones(length(labelDA1_test),1));
            idxhigh1 = labelDA1_test == "high risk";
            idxhigh2 = labelDA2_test == "high risk";
            idxhigh3 = labelDA3_test == "high risk";

            idxlow1 = labelDA1_test == "low risk";
            idxlow2 = labelDA2_test == "low risk";
            idxlow3 = labelDA3_test == "low risk";

            votehigh = sum([idxhigh1,idxhigh2,idxhigh3],2);
            votelow = sum([idxlow1,idxlow2,idxlow3],2);

            idxhighmaj = votehigh > votelow;
            labelDAfinal(idxhighmaj) = "high risk";
            labelDAfinal(not(idxhighmaj)) = "low risk";
            labelfinal = string(ones(length(labelDA1_test),1));




            % scoreModelSVM = fitSVMPosterior(modelSVM);
            [labelSVM_test,ScoreSVM_test] = predict(modelSVM,testset);

            rng('twister')
            [labelNN_test,ScoreNN_test] = predict(modelNN,testsetO);
            idxDAh = labelDAfinal == "high risk";
            idxSVMh = labelSVM_test == "high risk";
            idxNNh = labelNN_test == "high risk";

            idxDAl = labelDAfinal == "low risk";
            idxSVMl = labelSVM_test == "low risk";
            idxNNl = labelNN_test == "low risk";

            voteFinalH = sum([idxDAh,idxSVMh,idxNNh],2);
            voteFinalL = sum([idxSVMl,idxDAl,idxNNl],2);
            idxfinalH = voteFinalH>voteFinalL;
            labelfinal(idxfinalH) = "high risk";
            labelfinal(not(idxfinalH)) = "low risk";

           
            predicted_labels = labelfinal;
            

            rocObjDA1 = rocmetrics(ytest,ScoreDA1_test,modelDA1.ClassNames);
            auc_DA1 = rocObjDA1.AUC;

            rocObjDA2 = rocmetrics(ytest,ScoreDA2_test,modelDA2.ClassNames);
            auc_DA2 = rocObjDA2.AUC;

            rocObjDA3 = rocmetrics(ytest,ScoreDA3_test,modelDA3.ClassNames);
            auc_DA3 = rocObjDA3.AUC;

            rocObjSVM = rocmetrics(ytest,ScoreSVM_test,modelSVM.ClassNames);
            auc_SVM = rocObjSVM.AUC;

            rocObjNN = rocmetrics(ytest,ScoreNN_test,modelNN.ClassNames);
            auc_NN = rocObjNN.AUC;

            scores_ensemble = (auc_DA1(1)*ScoreDA1_test + auc_DA2(1)*ScoreDA2_test + auc_DA3(1)*ScoreDA3_test + auc_SVM(1)*ScoreSVM_test + auc_NN(1)*ScoreNN_test)/(auc_DA1(1) + auc_DA2(1) + auc_DA3(1) + auc_SVM(1) + auc_NN(1));
            probabilities = scores_ensemble;
            idx_ens_l = predicted_labels == "low risk";
            idx_ens_h = predicted_labels == "high risk";
            probabilities_final = zeros(length(probabilities),1);
            probabilities_final(idx_ens_l) = probabilities(idx_ens_l,2);
            probabilities_final(idx_ens_h) = probabilities(idx_ens_h,1);
            predictions = [obs, predicted_labels, probabilities_final];
        end



        
        function [TPR, TNR, PPV, FScore] = computeCMens(obj, cm, classNames)

            tp_m = diag(cm);
            for i = 1:size(classNames,1)
                TP = tp_m(i);
                FP = sum(cm(:, i), 1) - TP;
                FN = sum(cm(i, :), 2) - TP;
                TN = sum(cm(:)) - TP - FP - FN;

                TPR(i) = TP./(TP + FN);%tp/actual positive  RECALL SENSITIVITY
                if isnan(TPR(i))
                    TPR(i) = 0;
                end
                PPV(i) = TP./ (TP + FP); % tp / predicted positive PRECISION
                if isnan(PPV(i))
                    PPV(i) = 0;
                end
                TNR(i) = TN./ (TN+FP); %tn/ actual negative  SPECIFICITY
                if isnan(TNR(i))
                    TNR(i) = 0;
                end
                FPR(i) = FP./ (TN+FP);
                if isnan(FPR(i))
                    FPR(i) = 0;
                end
                FScore(i) = (2*(PPV(i) * TPR(i))) / (PPV(i)+TPR(i));

                if isnan(FScore(i))
                    FScore(i) = 0;
                end
            end

        end

        function displayResults(obj, performance, roc, predictions)
            disp(performance)
            figure
            plot(roc,ClassNames="high risk")
            disp(predictions)
            
        end

        function displayePredictions(obj, predictions)
            disp(predictions)
        end
        
    end
end