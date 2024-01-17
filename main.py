import argparse


from aillustrate.pipelines import SequentialPipeline, ParallelPipeline



if __name__ == '__main__':
    #python main.py --topic='Innovations and technologies' --concept_type='interior' --mode='parallel'
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--concept_type", type=str, required=True)
    parser.add_argument("--mode", type=str, required=False, default='parallel')
    args = parser.parse_args()
    topic, concept_type, mode = args.topic, args.concept_type, args.mode
    if mode == 'sequential':
        full_pipeline = SequentialPipeline(topic, concept_type)
    else:
        full_pipeline = ParallelPipeline(topic, concept_type)
    full_pipeline.run()


