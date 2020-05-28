cd /home/lsnoek1/spinoza_data/ni-edu/bids/code

FAILED = {};
main_dir = '../'; 
subs = dir(fullfile(main_dir, 'sub-*'));
for i=1:numel(subs)
    this_sub = fullfile(main_dir, subs(i).name);
    [~,sub_base,~] = fileparts(this_sub);
    sess = dir(fullfile(this_sub, 'ses-*'));
    for ii=1:numel(sess)
        this_ses = fullfile(this_sub, sess(ii).name);
        [~,ses_base,~] = fileparts(this_ses);
        physios = dir(fullfile(this_ses, 'func', '*physio.tsv.gz'));
        for iii=1:numel(physios)
           this_phys = fullfile(this_ses, 'func', physios(iii).name);
           this_func = strrep(this_phys,'recording-respcardiac_physio.tsv.gz', 'bold.nii.gz');
           save_dir = fullfile('../derivatives/physiology', sub_base, ses_base, 'physio');
           [~,basename,~] = fileparts(this_phys);
           ricor_out = strrep(basename, 'physio.tsv','desc-retroicor_regressors.tsv');

           if exist(fullfile(save_dir, ricor_out), 'file') ~= 2
               fprintf('Trying to create %s ...', ricor_out); 
               nii = load_untouch_header_only(this_func);
               n_slices = nii.dime.dim(2);
               n_vols = nii.dime.dim(5);
               tr = nii.dime.pixdim(5);
               try
                   run_retroicor(this_phys, save_dir, n_slices, n_vols, tr, 0, 0, 0);
               catch
                   fprintf('\n--------------\nFAILED!!! %s \n--------------\n\n', this_phys);
                   FAILED{end+1} = this_phys;
               end
           else
               fprintf('%s already exists ... skipping!\n', ricor_file);
           end
        end
    end
end
