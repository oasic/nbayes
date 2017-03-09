require File.expand_path(File.dirname(__FILE__) + '/spec_helper')
require 'fileutils'

describe "NBayes" do
  before do
    @nbayes = NBayes::Base.new
  end

  it "should assign equal probability to each class" do
    @nbayes.train( %w[a b c d e f g], 'classA' )
    @nbayes.train( %w[a b c d e f g], 'classB' )
    results = @nbayes.classify( %w[a b c] )
    expect(results['classA']).to eq(0.5)
    expect(results['classB']).to eq(0.5)
  end

  it "should handle more than 2 classes" do
    @nbayes.train( %w[a a a a], 'classA' )
    @nbayes.train( %w[b b b b], 'classB' )
    @nbayes.train( %w[c c], 'classC' )
    results = @nbayes.classify( %w[a a a a b c] )
    expect(results.max_class).to eq('classA')
    expect(results['classA']).to be >= 0.4
    expect(results['classB']).to be <= 0.3
    expect(results['classC']).to be <= 0.3
  end

  it "should use smoothing by default to eliminate errors w/division by zero" do
    @nbayes.train( %w[a a a a], 'classA' )
    @nbayes.train( %w[b b b b], 'classB' )
    results = @nbayes.classify( %w[x y z] )
    expect(results['classA']).to be >= 0.0
    expect(results['classB']).to be >= 0.0
  end

  it "should optionally purge low frequency data" do
    100.times do
      @nbayes.train( %w[a a a a], 'classA' )
      @nbayes.train( %w[b b b b], 'classB' )
    end
    @nbayes.train( %w[a], 'classA' )
    @nbayes.train( %w[c b], 'classB' )
    results = @nbayes.classify( %w[c] )
    expect(results.max_class).to eq('classB')
    expect(results['classB']).to be > 0.5
    expect(@nbayes.data.count_of_token_in_category('classB', 'c')).to eq(1)

    # this removes the entry for 'c' in 'classB' because it has freq of 1
    @nbayes.purge_less_than(2)
						# NOTE: this does not decrement the 'example' count
    results = @nbayes.classify( %w[c] )
    expect(@nbayes.data.count_of_token_in_category('classB', 'c')).to eq(0)
    expect(results['classA']).to eq(0.5)
    expect(results['classB']).to eq(0.5)
  end

  it "works on all tokens - not just strings" do
    @nbayes.train( [1, 2, 3], 'low' )
    @nbayes.train( [5, 6, 7], 'high' )
    results = @nbayes.classify( [2] )
    expect(results.max_class).to eq('low')
    results = @nbayes.classify( [6] )
    expect(results.max_class).to eq('high')
  end

  it "should optionally allow class distribution to be assumed uniform" do
    # before uniform distribution
    @nbayes.train( %w[a a a a b], 'classA' )
    @nbayes.train( %w[a a a a], 'classA' )
    @nbayes.train( %w[a a a a], 'classB' )
    results = @nbayes.classify( ['a'] )
    expect(results.max_class).to eq('classA')
    expect(results['classA']).to be > 0.5
    # after uniform distribution assumption
    @nbayes.assume_uniform = true
    results = @nbayes.classify( ['a'] )
    expect(results.max_class).to eq('classB')
    expect(results['classB']).to be > 0.5
  end

  it "should allow log of vocab size in smoothing" do

  end

  # In binarized mode, the frequency count is set to 1 for each token in each instance
  # For text, this is "set of words" rather than "bag of words"
  it "should allow binarized mode" do
    # w/o binarized mode, token repetition can skew the results
    def train_it
      @nbayes.train( %w[a a a a a a a a a a a], 'classA' )
      @nbayes.train( %w[b b], 'classA' )
      @nbayes.train( %w[a c], 'classB' )
      @nbayes.train( %w[a c], 'classB' )
      @nbayes.train( %w[a c], 'classB' )
    end
    train_it
    results = @nbayes.classify( ['a'] )
    expect(results.max_class).to eq('classA')
    expect(results['classA']).to be > 0.5
    # this does not happen in binarized mode
    @nbayes = NBayes::Base.new(:binarized => true)
    train_it
    results = @nbayes.classify( ['a'] )
    expect(results.max_class).to eq('classB')
    expect(results['classB']).to be > 0.5
  end

  it "allows smoothing constant k to be set to any value" do
    # increasing k increases smoothing
    @nbayes.train( %w[a a a c], 'classA' )
    @nbayes.train( %w[b b b d], 'classB' )
    expect(@nbayes.k).to eq(1)
    results = @nbayes.classify( ['c'] )
    prob_k1 = results['classA']
    @nbayes.k = 5
    results = @nbayes.classify( ['c'] )
    prob_k5 = results['classA']
    expect(prob_k1).to be > prob_k5 # increasing smoothing constant dampens the effect of the rare token 'c'
  end

  it "optionally allows using the log of vocab size during smoothing" do
    10_000.times do
      @nbayes.train( [rand(100)], 'classA' )
      @nbayes.train( %w[b b b d], 'classB' )
    end
  end

  describe "saving" do
    before do
      @tmp_dir = File.join( File.dirname(__FILE__), 'tmp')
      FileUtils.mkdir(@tmp_dir)  if !File.exists?(@tmp_dir)
      @yml_file = File.join(@tmp_dir, 'test.yml')
    end

    after do
      FileUtils.rm(@yml_file)  if File.exists?(@yml_file)
    end

    it "should save to yaml and load from yaml" do
      @nbayes.train( %w[a a a a], 'classA' )
      @nbayes.train( %w[b b b b], 'classB' )
      results = @nbayes.classify( ['b'] )
      expect(results['classB']).to be >= 0.5
      @nbayes.dump(@yml_file)
      expect(File.exists?(@yml_file)).to eq(true)
      @nbayes2 = NBayes::Base.from(@yml_file)
      results = @nbayes.classify( ['b'] )
      expect(results['classB']).to be >= 0.5
    end
  end

  it "should dump to yaml string and load from yaml string" do
    @nbayes.train( %w[a a a a], 'classA' )
    @nbayes.train( %w[b b b b], 'classB' )
    results = @nbayes.classify( ['b'] )
    expect(results['classB']).to be >= 0.5
    yml = @nbayes.dump(@nbayes)
    @nbayes2 = NBayes::Base.new.load(yml)
    results = @nbayes.classify( ['b'] )
    expect(results['classB']).to be >= 0.5
  end

  it "should delete a category" do
    @nbayes.train( %w[a a a a], 'classA' )
    @nbayes.train( %w[b b b b], 'classB' )
    expect(@nbayes.data.categories).to eq(["classA", "classB"])
    expect(@nbayes.delete_category('classB')).to eq(["classA"])
    expect(@nbayes.data.categories).to eq(["classA"])
  end

  it "should do nothing if asked to delete an inexistant category" do
    @nbayes.train( %w[a a a a], 'classA' )
    expect(@nbayes.data.categories).to eq(["classA"])
    expect(@nbayes.delete_category('classB')).to eq(["classA"])
    expect(@nbayes.data.categories).to eq(["classA"])
  end

  it "should untrain a class" do
    @nbayes.train( %w[a b c d e f g], 'classA' )
    @nbayes.train( %w[a b c d e f g], 'classB' )
    @nbayes.train( %w[a b c d e f g], 'classB' )
    @nbayes.untrain( %w[a b c d e f g], 'classB' )
    results = @nbayes.classify( %w[a b c] )
    expect(results['classA']).to eq(0.5)
    expect(results['classB']).to eq(0.5)
  end

  it "should remove the category when the only example is untrained" do
    @nbayes.train( %w[a b c d e f g], 'classA' )
    @nbayes.untrain( %w[a b c d e f g], 'classA' )
    expect(@nbayes.data.categories).to eq([])
  end

  it 'try untraining a non-existant category' do
    @nbayes.train( %w[a b c d e f g], 'classA' )
    @nbayes.train( %w[a b c d e f g], 'classB' )
    @nbayes.untrain( %w[a b c d e f g], 'classC' )
    expect(@nbayes.data.categories).to eq(['classA', 'classB'])
    results = @nbayes.classify( %w[a b c] )
    expect(results['classA']).to eq(0.5)
    expect(results['classB']).to eq(0.5)
  end
end
