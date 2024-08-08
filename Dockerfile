FROM --platform=linux/amd64 pytorch/pytorch
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy nnU-Net results folder into resources folder
# The required files for nnUNet inference are:
# resources/nnUNet_results/.../
# |-- plans.json
# |-- dataset.json
# |-- fold_0/
# |---- checkpoint_final.pth
# |-- fold_1/...
# |-- fold_2/...
# |-- fold_3/...
# |-- fold_4/...
COPY --chown=user:user new_resources /opt/app/new_resources
COPY --chown=user:user acoustic_2d_classif /opt/app/acoustic_2d_classif

COPY --chown=user:user requirements.txt /opt/app/
# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy the inference script, the postprocessing script and utils to the container
COPY --chown=user:user inference_2dclassifier.py /opt/app/
COPY --chown=user:user model_2d_nnunet.py /opt/app/
COPY --chown=user:user inference_utils.py /opt/app/
COPY --chown=user:user postprocess.py /opt/app/

ENTRYPOINT ["python", "inference_2dclassifier.py"]
