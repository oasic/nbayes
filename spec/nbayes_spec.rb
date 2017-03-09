require File.expand_path(File.dirname(__FILE__) + '/spec_helper')
require 'fileutils'

describe NBayes do
  let(:nbayes) { NBayes::Base.new }

  describe 'should assign equal probability to each class' do
    let(:results) { nbayes.classify(%w(a b c)) }

    before do
      nbayes.train(%w(a b c d e f g), 'classA')
      nbayes.train(%w(a b c d e f g), 'classB')
    end

    specify { expect(results['classA']).to eq(0.5) }
    specify { expect(results['classB']).to eq(0.5) }
  end

  describe 'should handle more than 2 classes' do
    let(:results) { nbayes.classify(%w(a a a a b c)) }

    before do
      nbayes.train(%w(a a a a), 'classA')
      nbayes.train(%w(b b b b), 'classB')
      nbayes.train(%w(c c), 'classC')
    end

    specify { expect(results.max_class).to eq('classA') }
    specify { expect(results['classA']).to be >= 0.4 }
    specify { expect(results['classB']).to be <= 0.3 }
    specify { expect(results['classC']).to be <= 0.3 }
  end

  describe 'should use smoothing by default to eliminate errors' do
    context 'when dividing by zero' do
      let(:results) { nbayes.classify(%w(x y z)) }

      before do
        nbayes.train(%w(a a a a), 'classA')
        nbayes.train(%w(b b b b), 'classB')
      end

      specify { expect(results['classA']).to be >= 0.0 }
      specify { expect(results['classB']).to be >= 0.0 }
    end
  end

  describe 'should optionally purge low frequency data' do
    let(:results) { nbayes.classify(%w(c)) }
    let(:token_count) { nbayes.data.count_of_token_in_category('classB', 'c') }

    before do
      100.times do
        nbayes.train(%w(a a a a), 'classA')
        nbayes.train(%w(b b b b), 'classB')
      end
      nbayes.train(%w(a), 'classA')
      nbayes.train(%w(c b), 'classB')
    end

    context 'before purge' do
      specify { expect(results.max_class).to eq('classB') }
      specify { expect(results['classB']).to be > 0.5 }
      specify { expect(token_count).to eq(1) }
    end

    context 'after purge' do
      before { nbayes.purge_less_than(2) }

      specify { expect(results['classA']).to eq(0.5) }
      specify { expect(results['classB']).to eq(0.5) }
      specify { expect(token_count).to be_zero }
    end
  end

  it 'works on all tokens - not just strings' do
    nbayes.train([1, 2, 3], 'low')
    nbayes.train([5, 6, 7], 'high')
    results = nbayes.classify([2])
    expect(results.max_class).to eq('low')
    results = nbayes.classify([6])
    expect(results.max_class).to eq('high')
  end

  describe 'should optionally allow class distribution to be assumed uniform' do
    context 'before uniform distribution' do
      let(:before_results) { nbayes.classify(['a']) }

      before do
        nbayes.train(%w(a a a a b), 'classA')
        nbayes.train(%w(a a a a), 'classA')
        nbayes.train(%w(a a a a), 'classB')
      end

      specify { expect(before_results.max_class).to eq('classA') }
      specify { expect(before_results['classA']).to be > 0.5 }

      context 'and after uniform distribution assumption' do
        let(:after_results) { nbayes.classify(['a']) }

        before { nbayes.assume_uniform = true }

        specify { expect(after_results.max_class).to eq('classB') }
        specify { expect(after_results['classB']).to be > 0.5 }
      end
    end
  end

  xit 'should allow log of vocab size in smoothing' do
  end

  # In binarized mode, the frequency count is set to 1 for each token in each instance
  # For text, this is "set of words" rather than "bag of words"
  it 'should allow binarized mode' do
    # w/o binarized mode, token repetition can skew the results
    # def train_it
    nbayes.train(%w(a a a a a a a a a a a), 'classA')
    nbayes.train(%w(b b), 'classA')
    nbayes.train(%w(a c), 'classB')
    nbayes.train(%w(a c), 'classB')
    nbayes.train(%w(a c), 'classB')
    # end
    # train_it
    results = nbayes.classify(['a'])
    expect(results.max_class).to eq('classA')
    expect(results['classA']).to be > 0.5
    # this does not happen in binarized mode
    nbayes = NBayes::Base.new(binarized: true)
    nbayes.train(%w(a a a a a a a a a a a), 'classA')
    nbayes.train(%w(b b), 'classA')
    nbayes.train(%w(a c), 'classB')
    nbayes.train(%w(a c), 'classB')
    nbayes.train(%w(a c), 'classB')
    results = nbayes.classify(['a'])
    expect(results.max_class).to eq('classB')
    expect(results['classB']).to be > 0.5
  end

  it 'allows smoothing constant k to be set to any value' do
    # increasing k increases smoothing
    nbayes.train(%w(a a a c), 'classA')
    nbayes.train(%w(b b b d), 'classB')
    expect(nbayes.k).to eq(1)
    results = nbayes.classify(['c'])
    prob_k1 = results['classA']
    nbayes.k = 5
    results = nbayes.classify(['c'])
    prob_k5 = results['classA']
    expect(prob_k1).to be > prob_k5 # increasing smoothing constant dampens the effect of the rare token 'c'
  end

  xit 'optionally allows using the log of vocab size during smoothing' do
    10_000.times do
      nbayes.train([rand(100)], 'classA')
      nbayes.train(%w(b b b d), 'classB')
    end
  end

  describe 'saving' do
    let(:tmp_dir) { File.join(File.dirname(__FILE__), 'tmp') }
    let(:yml_file) { File.join(tmp_dir, 'test.yml') }

    before { FileUtils.mkdir(tmp_dir) unless File.exist?(tmp_dir) }

    after { FileUtils.rm(yml_file) if File.exist?(yml_file) }

    it 'should save to yaml and load from yaml' do
      nbayes.train(%w(a a a a), 'classA')
      nbayes.train(%w(b b b b), 'classB')
      results = nbayes.classify(['b'])
      expect(results['classB']).to be >= 0.5
      nbayes.dump(yml_file)
      expect(File.exist?(yml_file)).to eq(true)
      nbayes2 = NBayes::Base.from(yml_file)
      results = nbayes.classify(['b'])
      expect(results['classB']).to be >= 0.5
    end
  end

  it 'should dump to yaml string and load from yaml string' do
    nbayes.train(%w(a a a a), 'classA')
    nbayes.train(%w(b b b b), 'classB')
    results = nbayes.classify(['b'])
    expect(results['classB']).to be >= 0.5
    yml = nbayes.dump(nbayes)
    nbayes2 = NBayes::Base.new.load(yml)
    results = nbayes.classify(['b'])
    expect(results['classB']).to be >= 0.5
  end

  describe 'should delete a category' do
    before do
      nbayes.train(%w(a a a a), 'classA')
      nbayes.train(%w(b b b b), 'classB')
      expect(nbayes.data.categories).to eq(%w(classA classB))
      expect(nbayes.delete_category('classB')).to eq(['classA'])
    end

    specify { expect(nbayes.data.categories).to eq(['classA']) }
  end

  describe 'should do nothing if asked to delete an inexistant category' do
    before { nbayes.train(%w(a a a a), 'classA') }

    specify { expect(nbayes.data.categories).to eq(['classA']) }
    specify { expect(nbayes.delete_category('classB')).to eq(['classA']) }
    specify { expect(nbayes.data.categories).to eq(['classA']) }
  end

  describe 'should untrain a class' do
    let(:results) { nbayes.classify(%w(a b c)) }

    before do
      nbayes.train(%w(a b c d e f g), 'classA')
      nbayes.train(%w(a b c d e f g), 'classB')
      nbayes.train(%w(a b c d e f g), 'classB')
      nbayes.untrain(%w(a b c d e f g), 'classB')
    end

    specify { expect(results['classA']).to eq(0.5) }
    specify { expect(results['classB']).to eq(0.5) }
  end

  describe 'should remove the category when the only example is untrained' do
    before do
      nbayes.train(%w(a b c d e f g), 'classA')
      nbayes.untrain(%w(a b c d e f g), 'classA')
    end

    specify { expect(nbayes.data.categories).to eq([]) }
  end

  describe 'try untraining a non-existant category' do
    let(:results) { nbayes.classify(%w(a b c)) }

    before do
      nbayes.train(%w(a b c d e f g), 'classA')
      nbayes.train(%w(a b c d e f g), 'classB')
      nbayes.untrain(%w(a b c d e f g), 'classC')
    end

    specify { expect(nbayes.data.categories).to eq(%w(classA classB)) }
    specify { expect(results['classA']).to eq(0.5) }
    specify { expect(results['classB']).to eq(0.5) }
  end
end
