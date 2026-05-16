FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install MLflow
RUN pip install --no-cache-dir mlflow==2.8.1 boto3

# Expose port
EXPOSE 5000

# Set entrypoint to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--default-artifact-root", "s3://ryam-mlflow-bucket-mlops/mlflow-artifacts"]
