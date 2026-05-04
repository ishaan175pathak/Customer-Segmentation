from __future__ import annotations
from zipfile import ZipFile
from pathlib import Path
from typing import Any, List, Tuple, TypeAlias, Literal
import pandas
from tqdm import tqdm
import argparse

class DataLoader:
    def __init__(self, /, indir: Path, inputFile: str | Path, outdir: Path | None = None) -> None:
        """
        It extracts or loads the dataset for Machine Learning model training

        Args:
            indir (Path): Contains the Location of the zipped file or CSV file for model training
            inputFile (String): Name of the file to be Extracted or Loaded
            outdir (Path): If the inputFile is a Zip File, then the location for extracting the File. \nDefaults to the current working directory.

        Returns:
            None
        """

        self.fileLocation: Path = indir / inputFile
        self.outdir: Path = Path(".") if not outdir else Path(outdir)
            

    def __extractFile__(self) -> pandas.DataFrame:
        """
            This Method Extracts the zip file to the mentioned location and returns the DataFrame if required

            Args:
                None

            Returns:
                (DataFrame or None): Returns DataFrame if the user requests the file otherwise None 

        """

        if self.fileLocation.suffix == ".zip":
            Path(self.outdir).mkdir(exist_ok=True)

            with ZipFile(self.fileLocation) as zf:
                total: int = sum(file.file_size for file in zf.infolist())

                with tqdm(total=total, unit="B", unit_scale=True, desc="Extracting") as pbar:
                    for member in zf.infolist():
                        zf.extract(member, path=self.outdir)
                        pbar.update(member.file_size)

        return pandas.read_csv(self.fileLocation)
        
    
if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    
    parser.add_argument("--indir", default=".")
    parser.add_argument("--fileName", type = str, default="archive (2).zip")
    parser.add_argument("--outdir", default=".")
    args: Any = parser.parse_args()

    indir: Path = Path(args.indir)
    outdir: Path = Path(args.outdir)

    obj: DataLoader = DataLoader(indir, args.fileName, outdir)
    