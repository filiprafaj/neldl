require 'torch'
require 'nn'
require 'os'
el_path = os.getenv("EL_PATH")
cmd = torch.CmdLine()
cmd:option('-ent_vecs_folder', '', 'the file path for the entity vectors with best score.')
cmd:option('-ent_vecs_file', '', 'the file path for the entity vectors with best score.')
opt = cmd:parse(arg or {})

ent_vecs = torch.load(opt.ent_vecs_folder..'/'..opt.ent_vecs_file)
print('number of entities: ' .. ent_vecs:size(1))
print('embeddings size: ' .. ent_vecs:size(2))

-- Normalize
ent_vecs = nn.Normalize(2):forward(ent_vecs:double())

-- print them to txt file
out = assert(io.open(opt.ent_vecs_folder.."/entity_embeddings.txt", "w")) -- open a file for serialization
splitter = " "
for i=1,ent_vecs:size(1) do
    for j=1,ent_vecs:size(2) do
        out:write(ent_vecs[i][j])
        if j == ent_vecs:size(2) then
            out:write("\n")
        else
            out:write(splitter)
        end
    end
end
out:close()


