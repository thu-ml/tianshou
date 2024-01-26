from tbparse import SummaryReader

def read_data(log_dir):
    reader = SummaryReader(log_dir)
    print(reader)
    df = reader.scalars
    print(df['tag'])


if __name__ == '__main__':
    log_dir = "examples/mujoco/log/Ant-v4/ppo/42/20240126-105848"
    read_data(log_dir)