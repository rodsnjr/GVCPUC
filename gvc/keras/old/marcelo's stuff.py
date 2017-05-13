    model.compile(loss=euclidean_distance, #MyLoss,#euclidean_distance,
                  optimizer=MyOptimizer, 
                  metrics=[MAE,MSE]) #,'mean_absolute_error''mean_squared_error', 

    model_json = model.to_json()
    with open(directoryModel+'model.json', "w") as json_file:
        json_file.write(model_json)


    plot(model1, to_file=directoryModel+'model1.png', show_shapes=True)  
    plot(model2, to_file=directoryModel+'model2.png', show_shapes=True)  
    plot(model3, to_file=directoryModel+'model3.png', show_shapes=True)  
    plot(model, to_file=directoryModel+'model.png', show_shapes=True)  

     checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto', save_weights_only=True)

    # para parar quando nao hover mudanca no val_loss
    earlyStopping= EarlyStopping(monitor='val_loss', patience=35, verbose=0, mode='auto')



      his = model.fit([X_train,X_train,X_train], Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True,
                verbose=1,
                validation_data=([X_test,X_test,X_test], Y_test),
                callbacks=callbacks_list)
