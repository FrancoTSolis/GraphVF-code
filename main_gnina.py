import os, shlex, sys, tempfile, pickle
from subprocess import Popen, PIPE

### config
epoch = 40
data_root='./data/crossdocked_pocket10'
path = './trained_model_ring_first_bond_latent_pocket10/gen_mols_epoch_{}'.format(epoch)

def get_temp_file():
    with tempfile.NamedTemporaryFile() as f:
        return f.name + '.sdf.gz'

def run_gnina(rec_file, lig_file, out_file):
    gnina = os.environ.get('GNINA_CMD', 'gnina')
    cmd = (
        f'{gnina} --minimize -r {rec_file} -l {lig_file} '
        f'--autobox_ligand {lig_file} -o {out_file}'
    )
    error = None
    last_stdout = ''
    proc = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    for c in iter(lambda: proc.stdout.read(1), b''): 
        sys.stdout.buffer.write(c)
        last_stdout += c.decode()
        if c == b'*' or c == b'\n': # progress bar or new line
            sys.stdout.flush()
            if last_stdout.startswith('WARNING'):
                error = last_stdout
            last_stdout = ''

    stderr = proc.stderr.read().decode()
    for stderr_line in stderr.split('\n'):
        if stderr_line.startswith('CUDNN Error'):
            error = stderr_line            

    print('GNINA STDERR', file=sys.stderr)
    print(stderr, file=sys.stderr)
    print('END GNINA STDERR', file=sys.stderr)

    return error, stderr

with open(path+'/global_index_to_rec_src.dict', 'rb') as f:
    rec_src = pickle.load(f)
with open(path+'/global_index_to_ref_lig_src.dict', 'rb') as f:
    lig_src = pickle.load(f)
assert len(rec_src) == len(lig_src)
print("{} pairs to score in total.".format(len(rec_src)))

for i in range(1, len(rec_src)+1):
    try:
        with open(path+"/{}.sdf".format(i)) as f: pass
    except EnvironmentError: # parent of IOError, OSError *and* WindowsError where available
        print("Invalid molecule: {}.sdf".format(i))
        continue
    
    rec_file = data_root+'/'+rec_src[i]
    lig_file = path+"/{}.sdf".format(i)
    out_file = get_temp_file()
    error, stderr = run_gnina(rec_file, lig_file, out_file)
