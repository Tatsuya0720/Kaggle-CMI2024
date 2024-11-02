from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def test_func():
    print('Hello from test_func')


default_args = {
    'owner': 'tatsuya',
    'start_date': datetime(2018, 10, 1),
}

with DAG(
    'ikeda-demo-dag', 
    default_args=default_args, 
    schedule_interval='0 18 * * *',
    catchup=False ) as dag:

  task1 = BashOperator(
      task_id='task1',
      bash_command='sleep 1'
  )

  task2 = BashOperator(
      task_id='task2',
      bash_command='sleep 2'
  )

  task3 = BashOperator(
      task_id='task3',
      bash_command='sleep 3'
  )

  task4 = PythonOperator(
      task_id='task4',
      python_callable=test_func
  )

  task5 = PythonOperator(
      task_id='task5',
      python_callable=test_func
  )

task1 >> task2 >> task3
task4 >> task3
task3 >> task5