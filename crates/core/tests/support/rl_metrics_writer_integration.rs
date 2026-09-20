use super::*;
use crate::{RlTrainerDriver, RlTrainerPhase, RlTrainerSeedContext, RlTrainerVessel};
use std::{cell::RefCell, rc::Rc};

struct Seeds;
impl RlTrainerSeedContext for Seeds {}
struct Vessel {
    counter: f64,
}
impl RlTrainerVessel for Vessel {
    type Seed = Seeds;
    type Environment = ();
    fn assign_trainer(&mut self, _: &Arc<RlTrainerRuntime>) -> Result<(), RlTrainerDriverError> {
        Ok(())
    }
    fn seeds(&mut self, _: RlTrainerPhase) -> Result<Seeds, RlTrainerDriverError> {
        Ok(Seeds)
    }
    fn environment(
        &mut self,
        _: &mut Seeds,
        _: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        Ok(())
    }
    fn run(
        &mut self,
        phase: RlTrainerPhase,
        (): &mut (),
        control: &mut RlTrainerControl,
    ) -> Result<(), RlTrainerDriverError> {
        if phase == RlTrainerPhase::Train {
            self.counter += 1.0;
            control.runtime.update(|s| {
                s.metrics = Some(IndexMap::from([
                    ("reward".into(), self.counter),
                    ("val/ignored".into(), 99.0),
                ]));
            })?;
        } else {
            control.runtime.update(|s| {
                s.metrics = Some(IndexMap::from([
                    ("val/reward".into(), self.counter / 2.0),
                    ("train/ignored".into(), -1.0),
                ]));
            })?;
        }
        Ok(())
    }
}
struct SharedWriter(Rc<RefCell<RlMetricsWriter>>);
impl RlTrainerCallback<Vessel> for SharedWriter {
    fn call(
        &mut self,
        hook: RlTrainerHook,
        control: &mut RlTrainerControl,
        vessel: &mut Vessel,
    ) -> Result<(), RlTrainerDriverError> {
        self.0.borrow_mut().call(hook, control, vessel)
    }
}

#[test]
fn real_trainer_writes_phase_files_preserves_reuse_and_leaves_test_hooks_inert() {
    let directory = tempfile::tempdir().unwrap();
    let train = directory.path().join("train_result.csv");
    let valid = directory.path().join("validation_result.csv");
    fs::write(&train, "preexisting train").unwrap();
    fs::write(&valid, "preexisting validation").unwrap();
    let writer = Rc::new(RefCell::new(
        RlMetricsWriter::new(directory.path().into(), CsvRlMetricsTableSink::default()).unwrap(),
    ));
    assert_eq!(fs::read_to_string(&train).unwrap(), "preexisting train");
    assert_eq!(
        fs::read_to_string(&valid).unwrap(),
        "preexisting validation"
    );
    let runtime = Arc::new(RlTrainerRuntime::new(None));
    let mut driver = RlTrainerDriver::new(
        Vessel { counter: 0.0 },
        runtime,
        crate::RlTrainerConfig {
            max_iters: Some(2.into()),
            val_every_n_iters: Some(1.into()),
        },
    );
    driver
        .callbacks
        .push(Box::new(SharedWriter(writer.clone())));
    driver.fit(None).unwrap();
    let train_text = fs::read_to_string(&train).unwrap();
    let valid_text = fs::read_to_string(&valid).unwrap();
    assert_eq!(train_text.replace("\r\n", "\n"), ",reward\n0,1.0\n1,2.0\n");
    assert_eq!(
        valid_text.replace("\r\n", "\n"),
        ",val/reward\n0,0.5\n1,1.0\n"
    );
    driver.test().unwrap();
    assert_eq!(fs::read_to_string(&train).unwrap(), train_text);
    assert_eq!(fs::read_to_string(&valid).unwrap(), valid_text);
    driver.fit(None).unwrap();
    assert_eq!(writer.borrow().train_records.len(), 4);
    assert_eq!(writer.borrow().valid_records.len(), 4);
    assert_eq!(
        writer.borrow().train_records[0]["reward"].to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        fs::read_to_string(train).unwrap().replace("\r\n", "\n"),
        ",reward\n0,1.0\n1,2.0\n2,3.0\n3,4.0\n"
    );
    let mut writer = writer.borrow_mut();
    let before = files(directory.path());
    writer.save_checkpoint().unwrap();
    writer.load_checkpoint(&()).unwrap();
    assert_eq!(writer.train_records.len(), 4);
    assert_eq!(files(directory.path()), before);
}
