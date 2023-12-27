import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from src.data import TeethDataModule
from src.model import CNNModel

from cfg import CFG


if __name__ == "__main__":
    # model 호출
    model = CNNModel()
    
    # data module -> data loader 들
    data_module = TeethDataModule()
    
    # call 함수
    checkpoint_callback = ModelCheckpoint(
        monitor='val/f1',
        mode="max",
        filename='teeth-{epoch:02d}-{val/f1:.2f}',
        save_last=True,
        every_n_epochs=1,
    )
    
    trainer = pl.Trainer(
        max_epochs=1000, 
        devices="auto",
        callbacks=[checkpoint_callback], 
        benchmark=True
    )
    trainer.fit(model=model, datamodule=data_module)

    trainer.test(model=model, datamodule=data_module)